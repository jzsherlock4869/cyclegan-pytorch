import cv2
import numpy as np
import albumentations.augmentations.functional as F

def save_tensor_as_imgs(tensor_list, save_fname):
    """
    tensor_list: each tensor have same size [1, 3, h, w], range 0 ~ 1
    tensor_list is 2d list, each sub-list in a row
    """
    for r_idx, sub_list in enumerate(tensor_list):
        for idx, im in enumerate(sub_list):
            im_np = im.detach().cpu().numpy()
            im_np = np.transpose(im_np[0,...], (1, 2, 0))
            if idx == 0:
                cat_im = im_np
            else:
                cat_im = np.concatenate((cat_im, im_np), axis=1)
        if r_idx == 0:
            all_im = cat_im
        else:
            all_im = np.concatenate((all_im, cat_im), axis=0)
    mean=(-0.485, -0.456, -0.406)
    std=(1/0.229, 1/0.224, 1/0.225)
    all_im = F.normalize(all_im, mean=mean, std=std, max_pixel_value=1.0)
    all_im = (np.clip(all_im, a_min=0, a_max=1) * 255.0).astype(np.uint8)
    cv2.imwrite(save_fname, all_im)






if __name__ == "__main__":
    from dataloader import get_photo2monet_eval_dataloader
    eval_loader = get_photo2monet_eval_dataloader('../dataset')
    save_ls = []
    for iter, batch in enumerate(eval_loader):
        if iter > 5:
            break
        save_ls = []
        iter_lsA = []
        iter_lsB = []
        imA, imB = batch["imgA"], batch["imgB"]
        iter_lsA.append(imA), iter_lsA.append(imA), iter_lsA.append(imA)
        iter_lsB.append(imB), iter_lsB.append(imB), iter_lsB.append(imB)
        save_ls.append(iter_lsA), save_ls.append(iter_lsB)
        save_tensor_as_imgs(save_ls, 'test_save_{}.png'.format(iter))
