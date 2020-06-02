import torch
from torch.utils.data import Dataset
from PIL import Image
from data_aug import *
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent = 4)

class BrainDataset(Dataset):
    def __init__(self, ids, annotations, transforms):
        self.ids = ids
        self.annotations = {a['filename'] : a for a in annotations }
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.ids[idx]
        if(img_path.find("/train/") != -1):
            mask_path = img_path.replace('train', 'train_mask').replace("jpg", "npy")
        else:
            mask_path = img_path.replace('val', 'val_mask').replace("jpg", "npy")
        #pp.pprint(self.annotations)
        mask_annotations = self.annotations[img_path.split("/")[-1]]

        img = Image.open(img_path).convert("RGB")
        mask = np.load(mask_path)

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        labels = torch.ones((num_objs, ), dtype = torch.int64)
        masks = torch.as_tensor(masks, dtype = torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs, ), dtype = torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

