import logging
import os
from collections import OrderedDict
import random

import torch
import torch.nn as nn

from utils.registry import MODEL_REGISTRY
from utils.vis_utils import save_tensor_as_imgs
from models.base_model import BaseModel

logger = logging.getLogger("base")


@MODEL_REGISTRY.register()
class CycleGANModel(BaseModel):
    """
    CycleGAN model which transfer images from diff domains in unsupervised manner
    Basic Architecture:
            -----netG_AB-------  fake B
          /                     \ netD_B
     domain A               domain B
  netD_A  \                     /
    fake A  ------netG_BA-------
    """
    def __init__(self, opt):
        super().__init__(opt)
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.data_names = ["real_A", "syn_B", "real_B", "syn_A"]
        self.network_names = ["netG_AB", "netG_BA", "netD_B", "netD_A"]
        self.networks = {}

        self.loss_names = [
            "adv_A",    # netG_BA
            "identity_A",    # netG_BA
            "cycle_AB",    # netG_AB + netG_BA
            "adv_B",    # netG_AB
            "identity_B",    # netG_AB
            "cycle_BA"   # netG_BA + netG_AB
        ]

        self.loss_weights = {}
        self.losses = {}
        self.optimizers = {}

        # define networks and load pretrained models
        # check network names correct
        nets_opt = opt["networks"]
        defined_network_names = list(nets_opt.keys())
        
        for name in defined_network_names:
            # same as : self.netG_AB = self.build_network(nets_opt['netG1']) (and others)
            setattr(self, name, self.build_network(nets_opt[name]))
            self.networks[name] = getattr(self, name)

        if self.is_train:
            # setup loss, optimizers, schedulers

            self.setup_train(opt["train"])

            self.max_grad_norm = opt["train"]["max_grad_norm"]
            self.D_ratio = opt["train"]["D_ratio"]

            ## buffer
            self.fake_A_buffer = ShuffleBuffer(opt["train"]["buffer_size"])
            self.fake_B_buffer = ShuffleBuffer(opt["train"]["buffer_size"])

    def feed_data(self, data):
        self.real_A = data["img_A"].to(self.device)
        self.real_B = data["img_B"].to(self.device)

    def forward_AtoB(self):
        # set .train() and .eval() to each network
        self.set_requires_grad(["netG_AB"], True)
        self.set_requires_grad(["netG_BA", "netD_B", "netD_A"], False)
        # discriminator forward
        self.fake_B = self.netG_AB(self.real_A)
        self.score_fake_B = self.netD_B(self.fake_B)
        # identity forward
        self.identity_B = self.netG_AB(self.real_B)
        # cycle forwards and backwards
        self.cycle_A = self.netG_BA(self.fake_B)
        self.cycle_B = self.netG_AB(self.netG_BA(self.identity_B))

    def forward_BtoA(self):
        # set .train() and .eval() to each network
        self.set_requires_grad(["netG_BA"], True)
        self.set_requires_grad(["netG_AB", "netD_A", "netD_B"], False)
        # discriminator forward
        self.fake_A = self.netG_BA(self.real_B)
        self.score_fake_A = self.netD_B(self.fake_A)
        # identity forward
        self.identity_A = self.netG_AB(self.real_A)
        # cycle forwards and backwards
        self.cycle_B = self.netG_AB(self.fake_A)
        self.cycle_A = self.netG_BA(self.netG_AB(self.identity_A))

    def optimize_parameters(self, step):

        loss_dict = OrderedDict()
        # ===================================================== #
        # forward netG_AB and calc loss, while other nets frozen
        loss_G_AB = 0
        self.forward_AtoB()
        # adv. loss for netG_AB in B domain
        if self.losses.get("adv_B"):
            self.set_requires_grad(["netD_B"], False)
            gab_adv_loss = self.calculate_gan_loss_G(
                self.netD_B, self.losses["adv_B"], self.fake_B
                )
            loss_dict["adv_B"] = gab_adv_loss.item()
            loss_G_AB += self.loss_weights["adv_B"] * gab_adv_loss
        
        # identity loss for netG_AB(B) and B
        if self.losses.get("identity_B"):
            gab_idt_loss = self.losses["identity_B"](self.identity_B, self.real_B)
            loss_dict["identity_B"] = gab_idt_loss.item()
            loss_G_AB += self.loss_weights["identity_B"] * gab_idt_loss

        # cycle loss for netG_BA(netG_AB(A)) and B, AND netG_AB(netG_BA(B)) and A
        if self.losses.get("cycle_AB"):
            gab_cycle_loss = self.losses["cycle_AB"](self.cycle_B, self.real_B) + \
                                self.losses["cycle_AB"](self.cycle_A, self.real_A)
            loss_dict["cycle_AB"] = gab_cycle_loss.item()
            loss_G_AB += self.loss_weights["cycle_AB"] * gab_cycle_loss

        self.set_optimizer(names=["netG_AB"], operation="zero_grad")
        loss_G_AB.backward()
        self.clip_grad_norm(names=["netG_AB"], norm=self.max_grad_norm)
        self.set_optimizer(names=["netG_AB"], operation="step")

        # ===================================================== #
        # forward netG_BA and calc loss, while other nets frozen
        loss_G_BA = 0
        self.forward_BtoA()
        # adv. loss for netG_BA in A domain
        if self.losses.get("adv_A"):
            self.set_requires_grad(["netD_A"], False)
            gba_adv_loss = self.calculate_gan_loss_G(
                self.netD_A, self.losses["adv_A"], self.fake_A
                )
            loss_dict["adv_A"] = gba_adv_loss.item()
            loss_G_BA += self.loss_weights["adv_A"] * gba_adv_loss

        # identity loss for netG_AB(B) and B
        if self.losses.get("identity_A"):
            gba_idt_loss = self.losses["identity_A"](self.identity_A, self.real_A)
            loss_dict["identity_A"] = gba_idt_loss.item()
            loss_G_BA += self.loss_weights["identity_A"] * gba_idt_loss

        # cycle loss for netG_BA(netG_AB(A)) and B, AND netG_AB(netG_BA(B)) and A
        if self.losses.get("cycle_BA"):
            gba_cycle_loss = self.losses["cycle_BA"](self.cycle_B, self.real_B) + \
                                self.losses["cycle_BA"](self.cycle_A, self.real_A)
            loss_dict["cycle_BA"] = gba_cycle_loss.item()
            loss_G_BA += self.loss_weights["cycle_BA"] * gba_cycle_loss

        self.set_optimizer(names=["netG_BA"], operation="zero_grad")
        loss_G_BA.backward()
        self.clip_grad_norm(names=["netG_BA"], norm=self.max_grad_norm)
        self.set_optimizer(names=["netG_BA"], operation="step")

        ## update netD_A, netD_B
        if self.losses.get("adv_B"):
            if step % self.D_ratio == 0:
                self.set_requires_grad(["netD_B"], True)
                loss_D_B = self.calculate_gan_loss_D(
                    self.netD_B, self.losses["adv_B"], self.real_B,
                    self.fake_B_buffer.choose(self.fake_B.detach())
                )
                loss_dict["d_adv_B"] = loss_D_B.item()
                loss_D_B = self.loss_weights["adv_B"] * loss_D_B

                self.set_optimizer(names=["netD_B"], operation="zero_grad")
                loss_D_B.backward()
                self.clip_grad_norm(["netD_B"], norm=self.max_grad_norm)
                self.set_optimizer(names=["netD_B"], operation="step")

        if self.losses.get("adv_A"):
            if step % self.D_ratio == 0:
                self.set_requires_grad(["netD_A"], True)
                loss_D_A = self.calculate_gan_loss_D(
                    self.netD_A, self.losses["adv_A"], self.real_A,
                    self.fake_A_buffer.choose(self.fake_A.detach())
                )
                loss_dict["d_adv_A"] = loss_D_A.item()
                loss_D_A = self.loss_weights["adv_A"] * loss_D_A

                self.set_optimizer(names=["netD_A"], operation="zero_grad")
                loss_D_A.backward()
                self.clip_grad_norm(["netD_A"], norm=self.max_grad_norm)
                self.set_optimizer(names=["netD_A"], operation="step")

        self.log_dict = loss_dict

    def calculate_gan_loss_D(self, netD, criterion, real, fake):
        d_pred_fake = netD(fake.detach())
        d_pred_real = netD(real)
        loss_real = criterion(d_pred_real, True, is_disc=True)
        loss_fake = criterion(d_pred_fake, False, is_disc=True)
        return (loss_real + loss_fake) / 2

    def calculate_gan_loss_G(self, netD, criterion, fake):
        d_pred_fake = netD(fake)
        loss_fake = criterion(d_pred_fake, True, is_disc=False)
        return loss_fake

    def save_current_visuals(self, save_dir, step):
        out_dict = OrderedDict()
        bsize = self.real_A.detach().size()[0]
        for bid in range(bsize):
            out_dict["img_A"] = self.real_A.detach()[bid: bid + 1].float().cpu()
            out_dict["fake_B"] = self.fake_B.detach()[bid: bid + 1].float().cpu()
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'iter{step}_im{bid}.png')
            save_tensor_as_imgs([[out_dict["img_A"], out_dict["fake_B"]]], save_path)
        return

    def test(self, data):
        self.real_A = data["img_A"].to(self.device)
        self.set_network_state(["netG_AB"], "eval")
        with torch.no_grad():
            self.fake_B = self.netG_AB(self.real_A)
        self.set_network_state(["netG_AB"], "train")


class ShuffleBuffer():
    """Random choose previous generated images or ones produced by the latest generators.
    :param buffer_size: the size of image buffer
    :type buffer_size: int
    """

    def __init__(self, buffer_size):
        """Initialize the ImagePool class.
        :param buffer_size: the size of image buffer
        :type buffer_size: int
        """
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.images = []

    def choose(self, images, prob=0.5):
        """Return an image from the pool.
        :param images: the latest generated images from the generator
        :type images: list
        :param prob: probability (0~1) of return previous images from buffer
        :type prob: float
        :return: Return images from the buffer
        :rtype: list
        """
        if self.buffer_size == 0:
            return  images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.buffer_size:
                self.images.append(image)
                return_images.append(image)
                self.num_imgs += 1
            else:
                p = random.uniform(0, 1)
                if p < prob:
                    idx = random.randint(0, self.buffer_size - 1)
                    stored_image = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(stored_image)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images