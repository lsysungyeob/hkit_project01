import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd

def clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def normalize(image, window_center, window_width):
    min_val = window_center - (window_width / 2)
    max_val = window_center + (window_width / 2)
    image_clipped = np.clip(image, min_val, max_val)
    image_normalized = (image_clipped - min_val) / (max_val - min_val + 1e-8)

    image_uint8 = (image_normalized * 255).astype(np.uint8)
    image_clahe = clahe(image_uint8)
    return image_clahe

def extract_roi(slice_img, slice_mask, margin, target_label):
    coords = np.where(slice_mask == target_label)
    if coords[0].size == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    y_min = max(0, y_min - margin)
    y_max = min(slice_img.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(slice_img.shape[1], x_max + margin)

    return slice_img[y_min:y_max, x_min:x_max]

def crop_or_pad(roi, out_size):
    h, w = roi.shape[:2]
    pad_h = max(out_size - h, 0)
    pad_w = max(out_size - w, 0)
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        roi = cv2.copyMakeBorder(
            roi,
            pad_top, pad_bottom,
            pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=0
        )
        h, w = roi.shape[:2]

    if h > out_size or w > out_size:
        cy, cx = h // 2, w // 2
        half = out_size // 2
        # roi = roi[cy-half:cy+half, cx-half:cx+half]
        y_start = cy - half
        y_end = y_start + out_size
        x_start = cx - half
        x_end = x_start + out_size
        roi = roi[y_start:y_end, x_start:x_end]

    return roi

class IDRIDataset(Dataset):
    def __init__(self, dataframe=None, transform=None, is_train=True, out_size=224):
        self.meta = dataframe
        self.transform = transform
        self.is_train = is_train
        self.out_size = out_size
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        slice = row['instance_number']
        score = row['malignancy']
        ww, wc = row['window_width'], row['window_center']
    
        img = sitk.GetArrayFromImage(sitk.ReadImage(row['image_path']))[slice]
        img = normalize(img, wc, ww)

        npz = np.load(row['label_path'], allow_pickle=True)
        slices = npz['slices']
        polygons = [np.array(p) for p in npz['polygons']]

        index = np.where(slices == slice)[0]
        mask = np.zeros_like(img, dtype=np.uint8)
        
        for i in index:
            pts = polygons[i][:, :2].astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], int(1))

        img = cv2.resize(img, (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.out_size, self.out_size), interpolation=cv2.INTER_NEAREST)

        if self.transform != None:
            if self.is_train:
                augmented = self.transform[0](image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]
                
                augmented = self.transform[1](image=img)
                img = augmented["image"]
                
                augmented = self.transform[2](image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]
            else:
                augmented = self.transform[0](image=img)
                img = augmented["image"]
                
                augmented = self.transform[1](image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]
        
        #img = torch.tensor(img, dtype=torch.float32)
        #mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = mask.unsqueeze(0).float()
        
        if score < 3: label = 0
        elif score > 3: label = 1
        else: pass

        return img, mask, label

MEAN = [0.5198, 0.5198, 0.5198]
STD  = [0.3159, 0.3159, 0.3159]

alb_train_transform = []
alb_train_transform.append(A.Compose([
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=0, p=0.5),
]))

alb_train_transform.append(A.Compose([
    A.ColorJitter(brightness=0.05, contrast=0.05, p=0.5),
    A.GaussNoise(var_limit=(0.0,0.01), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.CoarseDropout(max_holes=3, max_height=16, max_width=16, fill_value=0, p=0.3),
    A.Lambda(name="gray2rgb", image=lambda x, **kw: np.stack([x, x, x], axis=-1)),
    A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
]))

alb_train_transform.append(A.Compose([
    ToTensorV2(),
]))

alb_val_transform = []
alb_val_transform.append(A.Compose([
    A.Lambda(name="gray2rgb", image=lambda x, **kw: np.stack([x, x, x], axis=-1)),
    A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
]))

alb_val_transform.append(A.Compose([
    ToTensorV2(),
]))

def create_dataloader(batch_size=16):
    df = pd.read_csv('/data1/kerter/data.csv')
    df = df[df['malignancy'] != 3].reset_index(drop=True)
    
    #train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    split = gss.split(df, df['malignancy'], groups=df['subject_id'])

    train_idx, test_idx = next(split)

    train_df = df.iloc[train_idx]
    val_df = df.iloc[test_idx]
    
    print(len(train_df), len(val_df))

    train_dataset = IDRIDataset(train_df, transform=alb_train_transform, is_train=True)
    train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

    val_dataset = IDRIDataset(val_df, transform=alb_val_transform, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    return train_loader, val_loader

class IDRIPatchDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_train=True,  out_size=224, margin=30):
        self.meta      = dataframe
        self.transform = transform
        self.out_size  = out_size
        self.margin    = margin
        self.is_train = is_train

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        slice_idx = row['instance_number']
        score     = row['malignancy']
        ww, wc    = row['window_width'], row['window_center']

        if score <= 2:
            label = 0
        else:
            label = 1

        img  = sitk.ReadImage(row['image_path'])
        raw  = sitk.GetArrayFromImage(img)[slice_idx]
        norm = normalize(raw, wc, ww)

        data   = np.load(row['label_path'], allow_pickle=True)
        slices = data['slices']
        polys  = [np.array(p) for p in data['polygons']]
        mask   = np.zeros_like(norm, dtype=np.uint8)
        for i in np.where(slices == slice_idx)[0]:
            pts = polys[i][:, :2].astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], int(score))

        roi = extract_roi(norm, mask, self.margin, target_label=score)

        if roi is None:
            h, w = norm.shape
            cy, cx = h // 2, w // 2
            half = self.out_size // 2
            roi = norm[cy-half:cy+half, cx-half:cx+half]

        roi = crop_or_pad(roi, self.out_size)

        if self.transform != None:
            if self.is_train:
                augmented = self.transform[0](image=roi)
                img = augmented["image"]
                
                augmented = self.transform[1](image=img)
                img = augmented["image"]
                
                augmented = self.transform[2](image=img)
                img_tensor = augmented["image"]
            else:
                augmented = self.transform[0](image=roi)
                img = augmented["image"]
                
                augmented = self.transform[1](image=img)
                img_tensor = augmented["image"]

        return img_tensor, label
    
def create_dataloader_patch(batch_size=16, out_size=224):
    df = pd.read_csv('/data1/kerter/LIDC-IDRI-Dataset/data.csv')
    df = df[df['malignancy'] != 3].reset_index(drop=True)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # split = gss.split(df, df['malignancy'], groups=df['subject_id'])

    # train_idx, test_idx = next(split)

    # train_df = df.iloc[train_idx]
    # val_df = df.iloc[test_idx]
    
    print(len(train_df), len(val_df))

    train_dataset = IDRIPatchDataset(train_df, transform=alb_train_transform, is_train=True, out_size=out_size)
    train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

    val_dataset = IDRIPatchDataset(val_df, transform=alb_val_transform, is_train=False, out_size=out_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    return train_loader, val_loader