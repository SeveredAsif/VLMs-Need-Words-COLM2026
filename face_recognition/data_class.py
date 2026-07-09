from torch.utils.data import Dataset
import json
from PIL import Image

PROMPT = "Which person in the second image matches the person in the first image? Select from the following choices: A, B, C or D."

class VisionLanguageDataset(Dataset):
    def __init__(self, dataset_name):

        with open(f"{dataset_name}/annotations.json", "r") as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        ref_image_path = item["ref_image_path"]
        tgt_image_path = item["tgt_image_path"]
        
        ref_image = Image.open(ref_image_path).convert("RGB")
        tgt_image = Image.open(tgt_image_path).convert("RGB")

        target_images_order = item["metadata"]["target_images_order"]
            
        
        return {
            "prompt": PROMPT,
            "answer": item["answer"],
            "ref_image": ref_image,
            "tgt_image": tgt_image,
            "ref_coordinate": item["ref_coordinate"],
            "tgt_coordinate": item["tgt_coordinate"],
            "refer_person": item["metadata"]["ref_person_id"],
            "target_images_order": target_images_order
        }