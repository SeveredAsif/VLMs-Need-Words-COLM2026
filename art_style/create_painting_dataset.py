import os
import random
import json
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import argparse

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)

def resize_keep_aspect(img, max_size):
    ratio = min(max_size[0] / img.width, max_size[1] / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.Resampling.LANCZOS), ratio

def create_reference_image(image_path, label_height=60):
    img = Image.open(image_path).convert('RGB')
    img, _ = resize_keep_aspect(img, (400, 400)) # Resize for consistency
    canvas = Image.new('RGB', (img.width, img.height + label_height), color='white')
    canvas.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), "REF", font=font)
    x = (img.width - (bbox[2] - bbox[0])) // 2
    y = img.height + (label_height - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), "REF", fill='black', font=font)
    
    face_bbox = [[0, 0], [img.width, img.height]]
    return canvas, face_bbox

def create_target_image(image_paths, output_size=(400, 400), label_height=60, padding=20):
    labels = ['A', 'B', 'C', 'D']
    cell_w, cell_h = output_size[0], output_size[1] + label_height
    
    canvas = Image.new('RGB', (2 * cell_w + 3 * padding, 2 * cell_h + 3 * padding), color='white')
    draw = ImageDraw.Draw(canvas)
    
    positions = [
        (padding, padding),
        (padding + cell_w + padding, padding),
        (padding, padding + cell_h + padding),
        (padding + cell_w + padding, padding + cell_h + padding)
    ]
    
    face_bboxes = []
    
    for img_path, label, pos in zip(image_paths, labels, positions):
        img = Image.open(img_path).convert('RGB')
        img, _ = resize_keep_aspect(img, output_size)
        x_offset = pos[0] + (output_size[0] - img.width) // 2
        y_offset = pos[1] + (output_size[1] - img.height) // 2
        canvas.paste(img, (x_offset, y_offset))
        
        face_bbox = [[x_offset, y_offset], [x_offset + img.width, y_offset + img.height]]
        face_bboxes.append(face_bbox)
        
        bbox = draw.textbbox((0, 0), label, font=font_small)
        label_x = pos[0] + (cell_w - (bbox[2] - bbox[0])) // 2
        label_y = pos[1] + output_size[1] + (label_height - (bbox[3] - bbox[1])) // 2
        draw.text((label_x, label_y), label, fill='black', font=font_small)
    
    return canvas, face_bboxes

def generate_dataset(subset_df, output_dir, num_samples=1000):
    os.makedirs(os.path.join(output_dir, 'ref'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'target'), exist_ok=True)
    
    # Group by painter
    painter_images = {}
    for _, row in subset_df.iterrows():
        painter = row['painter_gt']
        img_path = os.path.join('paintings', row['filename'])
        if os.path.exists(img_path):
            if painter not in painter_images:
                painter_images[painter] = []
            painter_images[painter].append(img_path)
            
    # Filter painters with at least 2 images
    valid_painters = {p: imgs for p, imgs in painter_images.items() if len(imgs) >= 2}
    painter_ids = list(valid_painters.keys())
    
    if len(painter_ids) < 4:
        print(f"Not enough painters with >=2 images for {output_dir}. Only {len(painter_ids)} found.")
        return
        
    annotations = []
    labels = ['A', 'B', 'C', 'D']
    
    for idx in tqdm(range(num_samples), desc=f"Generating {output_dir}"):
        ref_painter = random.choice(painter_ids)
        ref_img_path, match_img_path = random.sample(valid_painters[ref_painter], 2)
        
        distractor_painters = random.sample([p for p in painter_ids if p != ref_painter], 3)
        distractor_images = [random.choice(valid_painters[p]) for p in distractor_painters]
        
        all_target_images = [match_img_path] + distractor_images
        random.shuffle(all_target_images)
        correct_answer = labels[all_target_images.index(match_img_path)]
        
        ref_output_path = os.path.join(output_dir, 'ref', f'{idx:06d}_ref.jpg')
        ref_canvas, ref_bbox = create_reference_image(ref_img_path)
        ref_canvas.save(ref_output_path, quality=95)
        
        tgt_output_path = os.path.join(output_dir, 'target', f'{idx:06d}_target.jpg')
        tgt_canvas, tgt_bboxes = create_target_image(all_target_images)
        tgt_canvas.save(tgt_output_path, quality=95)
        
        annotations.append({
            "id": idx,
            "ref_image_path": ref_output_path,
            "tgt_image_path": tgt_output_path,
            "question": "Which painting in the second image is by the same artist as the painting in the first image? Answer with A, B, C or D.",
            "options": ["A", "B", "C", "D"],
            "answer": correct_answer,
            "ref_coordinate": ref_bbox,
            "tgt_coordinate": tgt_bboxes,
            "metadata": {
                "ref_painter": ref_painter,
                "ref_original_image": ref_img_path,
                "match_original_image": match_img_path,
                "target_images_order": all_target_images,
                "distractor_painters": distractor_painters
            }
        })

    with open(os.path.join(output_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Created {num_samples} samples in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()

    df = pd.read_csv(f'{args.model_name}_painting_identification_results.csv')
    random.seed(42)
    
    print("Generating known dataset...")
    
    known_df = df[df['painter_correct'] == 'Yes']
    generate_dataset(known_df, f'{args.model_name}_known_paintings_dataset', num_samples=args.num_samples)
    
    print("Generating unknown dataset...")
    unknown_df = df[df['painter_correct'] == 'No']
    generate_dataset(unknown_df, f'{args.model_name}_unknown_paintings_dataset', num_samples=args.num_samples)
