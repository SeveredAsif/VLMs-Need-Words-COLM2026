import os
from torch.utils.data import Dataset
from PIL import Image
import json


PROMPT = 'Humans can find corresponding points for different objects in the same category. For instance, if there are images of two different cats, then the left ear tip of one cat corresponds to the left ear tip of the other cat, and the right front paw of one cat corresponds to the right front paw of the other cat.\nGiven the following two images, a reference point is annotated on the first image, labeled with REF. You are given multiple red-circled points on the second image, choices of "A, B, C, D" are drawn beside each circle. Select between the choices on the second image and find the corresponding point for the reference point. Which point is corresponding to the reference point?\nSelect from the following choices.\n(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D\n'

# PROMPT = '第二张图中哪个点最接近第一张图中的REF点？'

class VisionLanguageDataset(Dataset):
    def __init__(self, split="train", filter_conditions = None, box_size = 100):

        self.split = split
        self.box_size = box_size

        base_path = "SemCorVQA"

        if "og" in base_path:
            print("Using original dataset")

        if split == "train":
            with open(os.path.join(base_path, "train", "annotations.json"), "r") as f:
                self.data = json.load(f)
        elif split == "test":
            with open(os.path.join(base_path, "test", "annotations.json"), "r") as f:
                self.data = json.load(f)
        elif split == "val":
            with open(os.path.join(base_path, "val", "annotations.json"), "r") as f:
                self.data = json.load(f)
        elif split == "experiment":
            with open(os.path.join(base_path, "experiment", "annotations.json"), "r") as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Invalid split: {split}")

        keep = []
        if filter_conditions is not None:
            for item in self.data:
                belongs_flag = True
                for f in filter_conditions:
                    if f.startswith("NOT_"):
                        f = f[4:]
                        if f in item["kps_tags"]:
                            belongs_flag = False
                            break
                    else:
                        if f not in item["kps_tags"]:
                            belongs_flag = False
                            break
                if belongs_flag:
                    keep.append(item)
            print(f"Keeping {len(keep)} items out of {len(self.data)}")
            self.data = keep

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        ref_image = Image.open(item["ref_image_path"]).convert("RGB")
        tgt_image = Image.open(item["tgt_image_path"]).convert("RGB")
        

        ref_coordinate = item["ref_coordinate"]


        x1 = ref_coordinate[0] - self.box_size // 2
        y1 = ref_coordinate[1] - self.box_size // 2
        x2 = ref_coordinate[0] + self.box_size // 2
        y2 = ref_coordinate[1] + self.box_size // 2
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(ref_image.width, x2)
        y2 = min(ref_image.height, y2)
        ref_box = [(x1, y1), (x2, y2)]
        
        tgt_coordinates = item["corresponding_tgt"]

        x1 = tgt_coordinates[0] - self.box_size // 2
        y1 = tgt_coordinates[1] - self.box_size // 2
        x2 = tgt_coordinates[0] + self.box_size // 2
        y2 = tgt_coordinates[1] + self.box_size // 2
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(tgt_image.width, x2)
        y2 = min(tgt_image.height, y2)
        tgt_box = [(x1, y1), (x2, y2)]

        other_coordinates = item["tgt_points_sampled"]

        other_boxes = []
        for coordinate in other_coordinates:

            x1 = coordinate[0] - self.box_size // 2
            y1 = coordinate[1] - self.box_size // 2
            x2 = coordinate[0] + self.box_size // 2
            y2 = coordinate[1] + self.box_size // 2
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(tgt_image.width, x2)
            y2 = min(tgt_image.height, y2)
            other_boxes.append([(x1, y1), (x2, y2)])


        return {
            "prompt": PROMPT,
            "answer": item["answer"],
            "ref_image": ref_image,
            "tgt_image": tgt_image,
            "ref_box": ref_box,
            "tgt_box": tgt_box,
            "other_boxes": other_boxes,
            "og_src_img_path": item["og_src_image"],
            "og_tgt_img_path": item["og_tgt_image"],
        }




class VisionLanguageDataset2(Dataset):
    def __init__(self, split="train"):

        self.split = split

        if split == "train":
            with open("symbol_dataset_train_answers.json", "r") as f:
                self.data = json.load(f)
        elif split == "test":
            with open("symbol_dataset_test_answers.json", "r") as f:
                self.data = json.load(f)
        elif split == "val":
            with open("symbol_dataset_val_answers.json", "r") as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Invalid split: {split}")

        
    
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

