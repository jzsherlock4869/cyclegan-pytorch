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
    def __init__(self, root_dir, imgA_sub, imgB_sub, postfix_set=["png", "jpg"]):
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
            A.Resize(256, 256, cv2.INTER_CUBIC, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(p=1.0)],
            additional_targets={"image_2": 'image'})
        # self.transform = A.Compose([
        #     A.Resize(256, 256, cv2.INTER_CUBIC, p=1.0),
        #     A.HorizontalFlip(p=0.5)],
        #     additional_targets={"image_2": 'image'})
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
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        idA = idx % self.lenA
        idB = random.randint(0, self.lenB - 1)
        imgA = self.read_image("A", self.imgA_ids[idA])
        imgB = self.read_image("B", self.imgB_ids[idB])
        # imgA = self.read_image("A", self.imgA_ids[idA]) / 255.0
        # imgB = self.read_image("B", self.imgB_ids[idB]) / 255.0
        # imgA = self.transform(imgA)
        transformed = self.transform(image=imgA, image_2=imgB)
        imgA, imgB = transformed["image"], transformed["image_2"]
        imgA, imgB = imgA.transpose(2, 0, 1), imgB.transpose(2, 0, 1)
        return {"imgA": imgA, "imgB": imgB}

    def __len__(self):
        return min(self.lenA, self.lenB)

def get_photo2monet_train_dataloader(root_dir="../dataset", batch_size=4):
    imgA_sub, imgB_sub = "photo_jpg", "monet_jpg"
    postfix_set=["jpg"]
    train_dataset = CycleGANDataset(root_dir, imgA_sub, imgB_sub, postfix_set)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader

def get_photo2monet_eval_dataloader(root_dir="../dataset"):
    imgA_sub, imgB_sub = "photo_jpg", "monet_jpg"
    postfix_set=["jpg"]
    eval_dataset = CycleGANDataset(root_dir, imgA_sub, imgB_sub, postfix_set)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    return eval_dataloader

if __name__ == "__main__":

    # test dataloader works normally
    #data_path = r"E:\datasets\kaggle\20211021_im_something_a_painter"
    data_path = "../dataset/"
    train_dataloader = get_photo2monet_train_dataloader(data_path)
    for idx, i in enumerate(train_dataloader):
        if idx > 5:
            break
        print(i["imgA"].size(), i["imgB"].size())
