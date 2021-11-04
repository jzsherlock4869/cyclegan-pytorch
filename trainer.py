import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
import torch.backends.cudnn as cudnn

from datetime import datetime
from dataloader import get_photo2monet_train_dataloader, get_photo2monet_eval_dataloader
from networks.generators import Res_Generator
from networks.discriminators import VGG_Discriminator

def get_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def train_epoch(G_ab, G_ba, D_a, D_b, dataloader, optimizers, epoch_idx):

    G_ab.train(), G_ba.train(), D_a.train(), D_b.train()
    identity_loss = nn.L1Loss()
    cycle_loss = nn.L1Loss()
    gan_loss = nn.BCEWithLogitsLoss()
    a_id, a_cyc, a_gan = 0.5, 0.5, 0.2

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
        loss_tot = a_id * l_identity + a_cyc * l_cycle + a_gan * l_gan
        loss_tot.backward()
        for k in optimizers:
            optimizers[k].step()
        epoch_loss += loss_tot.item()
        print('Epoch [{}] Iter [{}/{}] || tot loss : {:.4f} (identity {:.4f}, cycle {:.4f}, gan {:.4f}) || timestamp {}'\
            .format(epoch_idx, iter_idx, len(dataloader), loss_tot.item(), l_identity.item(), l_cycle.item(), l_gan.item(), get_now()))
    
    return epoch_loss / len(dataloader)
        
def eval_epoch(G_ab, G_ba, D_a, D_b, dataloader, epoch_idx):
    G_ab.eval(), G_ba.eval(), D_a.eval(), D_b.eval()
    return

def save_epoch(G_ab, G_ba, D_a, D_b, epoch_idx):
    
    return

def main():

    """
    some settings
    """
    lr_set = {
        "G_ab": 1e-3,
        "G_ba": 1e-3,
        "D_a": 1e-4,
        "D_b": 1e-4
    }

    # data_path = r"E:\datasets\kaggle\20211021_im_something_a_painter"
    data_path = "../dataset/"
    eval_interval = 20
    save_interval = 20
    batch_size = 4
    use_gpu = [0]
    ##########################

    # define networks
    G_ab = Res_Generator() # generate B domain
    G_ba = Res_Generator() # generate A domain
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

    train_dataloader = get_photo2monet_train_dataloader(data_path, batch_size=batch_size)

    num_epoch = 50
    for epoch_idx in range(1, num_epoch + 1):
        epoch_loss = train_epoch(G_ab, G_ba, D_a, D_b, train_dataloader, optimizers, epoch_idx)
        print(" ===== Epoch {} completed, avg. tot. loss {:.4f}".format(epoch_idx, epoch_loss))
        
        if epoch_idx % eval_interval == 0:
            eval_dataloader = get_photo2monet_eval_dataloader(root_dir="../dataset", batch_size=1)
            eval_epoch(G_ab, G_ba, D_a, D_b, eval_dataloader, epoch_idx)

        if epoch_idx % save_interval == 0:
            save_epoch(G_ab, G_ba, D_a, D_b, epoch_idx)
    

if __name__ == "__main__":
    main()