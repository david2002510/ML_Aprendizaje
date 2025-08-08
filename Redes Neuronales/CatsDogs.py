import os
import pandas as pd 
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CatsDogsDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform= transforms.Compose([
            transforms.Resize((128,128)),
            transforms.CenterCrop(128),       # Recorta el centro para que quede cuadrado
            transforms.RandomHorizontalFlip(),      # Voltear horizontal aleatoriamente
            transforms.RandomRotation(15),          # Rotar hasta 15 grados aleatoriamente
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Cambios de color
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],    # Normaliza a [-1, 1]
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.annotations) 

    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = Image.open(img_path).convert("RGB")
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return(image,y_label)


