import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import torch.backends.cudnn as cudnn
torch.autograd.set_detect_anomaly(True)

import os

from datetime import datetime
from dataloader import get_photo2monet_train_dataloader, get_photo2monet_eval_dataloader
from networks.generators import Symm_ResDeconv_Generator, DnCNN_Generator
from networks.discriminators import VGG_Discriminator
from utils import save_tensor_as_imgs
    
def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def train_epoch(G_ab, G_ba, D_a, D_b, dataloader, optimizers, epoch_idx):

    G_ab.train(), G_ba.train(), D_a.train(), D_b.train()
    identity_loss = nn.L1Loss()
    cycle_loss = nn.L1Loss()
    gan_loss = nn.BCEWithLogitsLoss()
    # gan_loss = nn.MSELoss()
    # gan_loss = nn.BCELoss()
    a_id, a_cyc, a_gan = 1.0, 5.0, 50.0

    epoch_loss = 0
    for iter_idx, batch in enumerate(dataloader):
        for k in optimizers:
            optimizers[k].zero_grad()
        real_A, real_B = batch["imgA"], batch["imgB"]
        real_A, real_B = real_A.type(torch.cuda.FloatTensor), real_B.type(torch.cuda.FloatTensor)
        fake_B, fake_A = G_ab(real_A), G_ba(real_B)
        recon_A, recon_B = G_ba(fake_B), G_ab(fake_A)
        p_real_A, p_fake_A = D_a(real_A), D_a(fake_A)
        p_real_B, p_fake_B = D_b(real_B), D_b(fake_B)

        l_cycle = cycle_loss(recon_A, real_A) + cycle_loss(recon_B, real_B)
        l_identity = identity_loss(fake_B, real_A) + identity_loss(fake_A, real_B)
        l_gan = gan_loss(p_real_A, torch.ones_like(p_real_A)) + gan_loss(p_real_B, torch.ones_like(p_real_B)) \
            + gan_loss(p_fake_A, torch.zeros_like(p_fake_A)) + gan_loss(p_fake_B, torch.zeros_like(p_fake_B))
        l_gan = l_gan / 64.0
        loss_tot = a_id * l_identity + a_cyc * l_cycle + a_gan * l_gan
        # loss_tot = l_identity
        # loss_tot.backward(retain_graph=True)
        loss_tot.backward()
        for k in optimizers:
            optimizers[k].step()
        epoch_loss += loss_tot.detach().item()
        print('Epoch [{}] Iter [{}/{}] || tot loss : {:.4f} (identity {:.4f}, cycle {:.4f}, gan {:.4f}) || timestamp {}'\
            .format(epoch_idx, iter_idx, len(dataloader), loss_tot.item(), l_identity.item(), l_cycle.item(), l_gan.item(), get_now()))
        del loss_tot
        del fake_A, fake_B, recon_A, recon_B

    return epoch_loss / len(dataloader), optimizers

def eval_epoch(G_ab, G_ba, dataloader, epoch_idx, num_imgs, save_dir, verbose=True, log_interval=10):
    print('==== Start eval in Epoch {} ===='.format(epoch_idx))
    G_ab.eval(), G_ba.eval()
    for iter_idx, batch in enumerate(dataloader):
        show_ls = []
        if iter_idx == num_imgs:
            break
        if verbose and iter_idx % log_interval == 0:
            print(' >>> eval image no. {}'.format(iter_idx))
        real_A, real_B = batch["imgA"], batch["imgB"]
        real_A, real_B = real_A.type(torch.cuda.FloatTensor), real_B.type(torch.cuda.FloatTensor)
        fake_B, fake_A = G_ab(real_A), G_ba(real_B)
        recon_A, recon_B = G_ba(fake_B), G_ab(fake_A)
        show_ls.append([real_A, fake_B, recon_A])
        show_ls.append([real_B, fake_A, recon_B])
        os.makedirs(os.path.join(save_dir, 'epoch_{}'.format(epoch_idx)), exist_ok=True)
        save_tensor_as_imgs(show_ls, save_fname=os.path.join(save_dir, \
                                        'epoch_{}'.format(epoch_idx), \
                                        'test_epoch{}_batch_{}.png'.format(epoch_idx, iter_idx)))

    return

def save_epoch(G_ab, G_ba, D_a, D_b, epoch_idx):
    
    return


def main():

    """
    some settings
    """
    lr_set = {
        "G_ab": 1e-4,
        "G_ba": 1e-4,
        "D_a": 1e-6,
        "D_b": 1e-6
    }

    # data_path = r"E:\datasets\kaggle\20211021_im_something_a_painter"
    data_path = "../dataset/"
    eval_interval = 2
    save_interval = 20
    batch_size = 16
    use_gpu = [0]
    eval_num_imgs = 100
    eval_save_dir = './eval_output'
    num_epoch = 5000
    ##########################

    # define networks
    G_ab = Symm_ResDeconv_Generator() # generate B domain
    G_ba = Symm_ResDeconv_Generator() # generate A domain
    # G_ab = DnCNN_Generator()
    # G_ba = DnCNN_Generator()
    D_a = VGG_Discriminator() # check if in domain A
    D_b = VGG_Discriminator() # check if in domain B

    if use_gpu is not None:
        G_ab, G_ba, D_a, D_b = G_ab.cuda(), G_ba.cuda(), D_a.cuda(), D_b.cuda()

    if use_gpu is not None and len(use_gpu) > 1:
        G_ab = nn.DataParallel(G_ab)
        G_ba = nn.DataParallel(G_ba)
        D_a = nn.DataParallel(D_a)
        D_b = nn.DataParallel(D_b)

    # get optimizers
    optimizer_G_ab = optim.Adam(G_ab.parameters(), lr_set["G_ab"])
    optimizer_G_ba = optim.Adam(G_ba.parameters(), lr_set["G_ba"])
    optimizer_D_a = optim.Adam(D_a.parameters(), lr_set["D_a"])
    optimizer_D_b = optim.Adam(D_b.parameters(), lr_set["D_b"])

    optimizers = {
        "G_ab": optimizer_G_ab,
        "G_ba": optimizer_G_ba,
        "D_a": optimizer_D_a,
        "D_b": optimizer_D_b
    }

    # lr_sch_G_ab = optim.lr_scheduler.ExponentialLR(optimizers["G_ab"])

    train_dataloader = get_photo2monet_train_dataloader(data_path, batch_size=batch_size)

    for epoch_idx in range(1, num_epoch + 1):

        epoch_loss, optimizers = train_epoch(G_ab, G_ba, D_a, D_b, train_dataloader, optimizers, epoch_idx)
        print(" ===== Epoch {} completed, avg. tot. loss {:.4f}".format(epoch_idx, epoch_loss))
        
        if epoch_idx % eval_interval == 0:
            eval_dataloader = get_photo2monet_eval_dataloader(root_dir="../dataset")
            eval_epoch(G_ab, G_ba, eval_dataloader, epoch_idx, eval_num_imgs, eval_save_dir)

        if epoch_idx % save_interval == 0:
            save_epoch(G_ab, G_ba, D_a, D_b, epoch_idx)


if __name__ == "__main__":
    main()