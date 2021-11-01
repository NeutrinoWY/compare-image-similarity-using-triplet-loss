from torch.utils.data import Dataset
import os
import random
from PIL import Image
from torchvision import transforms
import torch

# Fix random seed
random.seed(37)

class ProjDataset(Dataset):
    
    def __init__(self, which, transform=None):
        self.which = which
        self.transform = transform

        if self.which in  ["train", "val"]:
            if not (os.path.isfile("train_split_data.txt") and os.path.isfile("val_split_data.txt")):
                print("Making a tyrain.val split which is going to be saved to txt files")
                with open("train_triplets.txt", "r") as fh:
                    data = fh.read().splitlines()

                img_names = os.listdir("food")

                data = sorted(data)
                img_names = sorted(img_names)

                # Randomly shuffle data
                random.shuffle(data)
                random.shuffle(img_names)
                random.shuffle(data)
                random.shuffle(img_names)
                random.shuffle(data)
                random.shuffle(img_names)

                # List of lists
                data = [[x for x in el.split()] for el in data]

                img_names = [el.split(".")[0] for el in img_names]

                val_id = []
                for img_num, image in enumerate(img_names):
                    print(f"{img_num}\{len(img_names)}, {len(val_id)}\{len(data)} \n")
                    if len(val_id) >= 0.1*len(data):
                        break
                    for i in range(len(data)):
                        if image in data[i]:  # TODO Maybe compare to just the first element (anchor)
                            if i not in val_id:
                                val_id.append(i)
                            continue
                # Save the train split
                train_data = [el for idx, el in enumerate(data) if idx not in val_id]
                with open("train_split_data.txt", "w") as fh:
                    for el in train_data:
                        fh.write(f"{el[0]} {el[1]} {el[2]}\n")
                # Save the val split
                val_data = [el for idx, el in enumerate(data) if idx in val_id]
                with open("val_split_data.txt", "w") as fh:
                    for el in val_data:
                        fh.write(f"{el[0]} {el[1]} {el[2]}\n")
                
            if self.which == "train":
                with open(os.path.join("train_split_data.txt"), "r") as fh:
                    self.data = fh.read().splitlines()
                
            elif self.which == "val":
                with open(os.path.join("val_split_data.txt"), "r") as fh:
                    self.data = fh.read().splitlines()
                
        elif self.which == "train_all":
            with open("train_triplets.txt", "r") as fh:
                self.data = fh.read().splitlines()

        elif self.which == "test":
            with open("test_triplets.txt", "r") as fh:
                self.data = fh.read().splitlines()
                
        else:
            raise AssertionError
        
        self.data = [[x for x in el.split()] for el in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        A_pth = os.path.join("food", f"{self.data[idx][0]}.jpg")
        A_img = Image.open(A_pth)

        B_pth = os.path.join("food", f"{self.data[idx][1]}.jpg")
        B_img = Image.open(B_pth)
        
        C_pth = os.path.join("food", f"{self.data[idx][2]}.jpg")
        C_img = Image.open(C_pth)

        if self.transform is not None:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)
            C_img = self.transform(C_img)
        
        return A_img, B_img, C_img
