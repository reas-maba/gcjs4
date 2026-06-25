import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = {}
        self.captions = {}
        
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            
            if image_id not in self.captions:
                self.captions[image_id] = []
            self.captions[image_id].append(caption)
        
        for img in self.annotations['images']:
            image_id = img['id']
            file_name = img['file_name']
            self.images[image_id] = file_name
        
        self.image_ids = list(self.captions.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_name = self.images[image_id]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        caption = self.captions[image_id][0]
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption
