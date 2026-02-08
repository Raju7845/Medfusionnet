import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer

class MedMultimodalDataset(Dataset):
    def __init__(self, data_list, transform=None, tokenizer_name="distilbert-base-uncased", max_len=128):
        self.data = data_list # List of (image_path, text, label, task_id)
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, text, label, task_id = self.data[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        inputs = self.tokenizer.encode_plus(
            text, padding='max_length', truncation=True, 
            max_length=self.max_len, return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'task_id': task_id # 0: BC, 1: CC, 2: PCOS
        }