import torch
import os
from glob import glob
from functools import reduce
import os.path as osp
import cv2
import random
import albumentations as A
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


class CycleGANDataset(Dataset):
    """
    cyclegan input style dataset
    two domain images are stored in two sub-dirs in the dataroot
    """
    def __init__(self, root_dir, imgA_sub, imgB_sub, postfix_set=["png", "jpg"], img_size=(256, 256)):
        """
        Args:
            root_dir: dataset root dir
            imgA_sub: image sub-dir name for domain A
            imgB_sub: image sub-dir name for domain B
            postfix_set: postfix to be scanned
            img_size: target size to resize the original images
        """
        imgA_path = osp.join(root_dir, imgA_sub)
        imgB_path = osp.join(root_dir, imgB_sub)
        imgA_lists = [glob(osp.join(imgA_path, "*." + postfix)) for postfix in postfix_set]
        imgB_lists = [glob(osp.join(imgB_path, "*." + postfix)) for postfix in postfix_set]
        imgA_ids = [osp.basename(i) for i in reduce(lambda x,y: x+y, imgA_lists)]
        imgB_ids = [osp.basename(i) for i in reduce(lambda x,y: x+y, imgB_lists)]
        self.imgA_path, self.imgB_path = imgA_path, imgB_path
        self.imgA_ids, self.imgB_ids = imgA_ids, imgB_ids
        self.lenA, self.lenB = len(self.imgA_ids), len(self.imgB_ids)
        self.transform = A.Compose([
            A.Resize(img_size[0], img_size[1], cv2.INTER_CUBIC, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(p=1.0, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0)],
            additional_targets={"image_2": 'image'})
        print("Dataset loaded from {} ...".format(root_dir))
        print("Domain A (len={}): {}, Domain B (len={}): {}"\
            .format(self.lenA, imgA_sub, self.lenB, imgB_sub))
    
    def read_image(self, AorB, image_id):
        if AorB == "A":
            dir_name = self.imgA_path
        else:
            assert AorB == "B"
            dir_name = self.imgB_path
        img = cv2.imread(osp.join(dir_name, image_id))
        return img

    def __getitem__(self, idx):
        idA = idx % self.lenA
        idB = random.randint(0, self.lenB - 1)
        imgA = self.read_image("A", self.imgA_ids[idA])
        imgB = self.read_image("B", self.imgB_ids[idB])
        transformed = self.transform(image=imgA, image_2=imgB)
        imgA, imgB = transformed["image"], transformed["image_2"]
        imgA, imgB = imgA.transpose(2, 0, 1), imgB.transpose(2, 0, 1)
        return {"img_A": imgA, "img_B": imgB}

    def __len__(self):
        return min(self.lenA, self.lenB)

def get_photo2monet_train_dataloader(root_dir="../datasets/monet_dataset", batch_size=8, img_size=(256, 256)):
    imgA_sub, imgB_sub = "photo_jpg", "monet_jpg"
    postfix_set=["jpg"]
    train_dataset = CycleGANDataset(root_dir, imgA_sub, imgB_sub, postfix_set, img_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader

def get_horse2zebra_train_dataloader(root_dir="../datasets/zebra_dataset", batch_size=4, img_size=(256, 256)):
    imgA_sub, imgB_sub = "trainA", "trainB"
    postfix_set=["jpg"]
    train_dataset = CycleGANDataset(root_dir, imgA_sub, imgB_sub, postfix_set, img_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader


if __name__ == "__main__":

    # test dataloader works normally
    data_path = "../datasets/monet_dataset"
    train_dataloader = get_photo2monet_train_dataloader(data_path)
    for idx, batch in enumerate(train_dataloader):
        if idx > 5:
            break
        print(batch["img_A"].size(), batch["img_B"].size())
