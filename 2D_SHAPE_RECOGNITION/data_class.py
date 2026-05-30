import os
from torch.utils.data import Dataset
from PIL import Image
import json

PROMPT = 'Which shape in the second image is most similar to the REF shape in the first image?\nSelect from the following choices.\n(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D\n'


class VisionLanguageDataset(Dataset):
    def __init__(self, dataset):
        with open(f"{dataset}/answers.json", "r") as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        ref_image_path = item["ref_image_path"]
        tgt_image_path = item["tgt_image_path"]
        
        ref_positions = item["ref_pixel_positions"]
        tgt_positions = item["tgt_pixel_positions"]

        ref_image = Image.open(ref_image_path).convert("RGB")
        tgt_image = Image.open(tgt_image_path).convert("RGB")


        
        return {
            "prompt": PROMPT,
            "answer": item["answer"],
            "ref_image": ref_image,
            "tgt_image": tgt_image,
            "ref_positions": ref_positions,
            "tgt_positions": tgt_positions,
            "shape_types": item["shape_types"],
        }

