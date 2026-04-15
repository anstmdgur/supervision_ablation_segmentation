import torch
from torch.utils.data import DataLoader, Dataset, Subset
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import os
import natsort

#CLAHE
#external 데이터를 활용한다면 A.randomBrightnessContrast를 활용해 밝기와 명암을 무작위 조절하여 일반화 성능을 높여볼 수 있음.

class ARCADE_train_dataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = A.Compose([
            A.Affine( #Rotate -15 ~ 15
                rotate=(-15, 15),
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1), 
                border_mode=cv2.BORDER_CONSTANT, #생긴 빈 공간을 하나의 값으로 채움
                fill=0, #빈공간을 0으로(검은색) 채움
                fill_mask=0,
                p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5),
            A.CLAHE(
                clip_limit=2.0, 
                tile_grid_size=(8, 8), #이미지를 8*8타일로 나눠서 생각함
                p=1.0),
            A.Normalize(
                mean=(0.449,), std=(0.226,), #imageNet의 평균과 표준편차
                max_pixel_value=255.0,
                p=1.0)], additional_targets={'mask': 'mask'}) #{'내가_쓸_이름': '알부멘테이션이_알아들을_타입'} 지워도 상관x
        self.image_filenames = natsort.natsorted(os.listdir(image_dir))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = mask / 255.0
        
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)    # (1, H, W)
        
        return image_tensor, mask_tensor
    
class XCAD_train_dataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                rotate=(-20, 20),
                scale=(0.85, 1.15),
                translate_percent=(-0.1, 0.1), 
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5),
            A.ElasticTransform( 
                alpha=120, 
                sigma=120 * 0.05, 
                alpha_affine=120 * 0.03, 
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5),
            A.GaussNoise( 
                var_limit=(10.0, 50.0), 
                p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5),
            A.CLAHE(
                clip_limit=2.0, 
                tile_grid_size=(8, 8), #이미지를 8*8타일로 나눠서 생각함
                p=1.0),
            A.Normalize(
                mean=(0.449,), std=(0.226,), #imageNet의 평균과 표준편차
                max_pixel_value=255.0,
                p=1.0)], additional_targets={'mask': 'mask'}) #{'내가_쓸_이름': '알부멘테이션이_알아들을_타입'} 지워도 상관x
        self.image_filenames = natsort.natsorted(os.listdir(image_dir))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = mask / 255.0
        
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)    # (1, H, W)
        
        return image_tensor, mask_tensor
    
class ARCADE_eval_dataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = A.Compose([
            A.CLAHE(
                clip_limit=2.0, 
                tile_grid_size=(8, 8),
                p=1.0),
            A.Normalize(
                mean=(0.449,), std=(0.226,), #imageNet의 평균과 표준편차
                max_pixel_value=255.0,
                p=1.0)], additional_targets={'mask': 'mask'})
        self.image_filenames = natsort.natsorted(os.listdir(image_dir))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        mask = mask / 255.0
        
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)    # (1, H, W)
        
        return image_tensor, mask_tensor
    
# class ARCADE_test_dataset(Dataset):
#     def __init__(self, image_dir, mask_dir):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.transform = A.Compose([
#             A.Normalize(
#                 mean=(0.449,), std=(0.226,), #imageNet의 평균과 표준편차
#                 max_pixel_value=255.0,
#                 p=1.0)], additional_targets={'mask': 'mask'})
#         self.image_filenames = natsort.natsorted(os.listdir(image_dir))
    
#     def __len__(self):
#         return len(self.image_filenames)
    
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_filenames[idx])
#         mask_path = os.path.join(self.mask_dir, self.image_filenames[idx])
        
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
#         if self.transform is not None:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
        
#         mask = mask / 255.0
        
#         image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, H, W)
#         mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)    # (1, H, W)
        
#         return image_tensor, mask_tensor
    
def get_ARCADE_loaders(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, test_image_dir, test_mask_dir, batch_size=8):
    train_dataset = ARCADE_train_dataset(train_image_dir, train_mask_dir)
    val_dataset = ARCADE_eval_dataset(val_image_dir, val_mask_dir)
    test_dataset = ARCADE_eval_dataset(test_image_dir, test_mask_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_XCAD_loaders(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, test_image_dir, test_mask_dir, batch_size=8):
    train_dataset = XCAD_train_dataset(train_image_dir, train_mask_dir)
    val_dataset = ARCADE_eval_dataset(val_image_dir, val_mask_dir)
    test_dataset = ARCADE_eval_dataset(test_image_dir, test_mask_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader



image_tensor, mask_tensor = ARCADE_eval_dataset('./data/XCAD/test/images/', 
                                                './data/XCAD/test/masks/')[1]
img_show = image_tensor.squeeze().numpy()
mask_show = mask_tensor.squeeze().numpy()
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi = 200)
axes[0].imshow(img_show, cmap='gray')
axes[0].set_title(f'Image')

axes[1].imshow(mask_show, cmap='gray')
axes[1].set_title(f'Ground Truth')

plt.savefig(f'./segmentation_framework/supervision_model_input.png', bbox_inches='tight')
plt.close(fig)