import argparse
import os
import math
import pickle
import random

parser = argparse.ArgumentParser()
parser.add_argument('--box_size', type=int, required=True, help='Box size around keypoint')
parser.add_argument('--filters', nargs='+', required=True, help='List of filters')
parser.add_argument('--model_path', type=str, required=True, help='Model path')
parser.add_argument('--cuda', type=int, required=True, help='CUDA device index')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from torch.utils.data import Subset
from data_classes import VisionLanguageDataset
from PIL import Image
import matplotlib.pyplot as plt


test_dataset = VisionLanguageDataset(split="test", box_size=args.box_size, filter_conditions=args.filters)
print(f"Using filter condition: {str(args.filters)} and box size: {args.box_size}")
print(f"Test dataset size: {len(test_dataset)}")

RESULTS_DIR = "spair_rep_results/"
OUTPUT_FILE = f"{RESULTS_DIR}/correct_per_layer_box{args.box_size}_filters{str(args.filters)}_model{args.model_path.replace('/', '_')}"
os.makedirs(RESULTS_DIR, exist_ok=True)

SUBSAMPLE_TEST_SIZE = 1000
SEED = 42

random.seed(SEED)
num_samples = min(SUBSAMPLE_TEST_SIZE, len(test_dataset))
indices = random.sample(range(len(test_dataset)), num_samples)
test_dataset = Subset(test_dataset, indices)

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left', do_pan_and_scan=True)
model = Gemma3ForConditionalGeneration.from_pretrained(
    args.model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
).cuda()

model.eval()

N_LAYERS = len(model.model.language_model.layers) + 1

print(f"Number of layers: {N_LAYERS}")

VISION_START_ID = 255999


def get_pan_and_scan_layout(width, height, processor):
    ip = processor.image_processor
    min_crop_size = getattr(ip, "pan_and_scan_min_crop_size", 256)
    if min_crop_size is None:
        min_crop_size = 256
    max_num_crops = getattr(ip, "pan_and_scan_max_num_crops", 4)
    if max_num_crops is None:
        max_num_crops = 4
    min_ratio = getattr(ip, "pan_and_scan_min_ratio_to_activate", 1.2)
    if min_ratio is None:
        min_ratio = 1.2

    if width >= height:
        if width / height < min_ratio:
            return 1, 1
        num_crops_w = int(math.floor(width / height + 0.5))
        num_crops_w = min(int(math.floor(width / min_crop_size)), num_crops_w)
        num_crops_w = max(2, num_crops_w)
        num_crops_w = min(max_num_crops, num_crops_w)
        num_crops_h = 1
    else:
        if height / width < min_ratio:
            return 1, 1
        num_crops_h = int(math.floor(height / width + 0.5))
        num_crops_h = min(int(math.floor(height / min_crop_size)), num_crops_h)
        num_crops_h = max(2, num_crops_h)
        num_crops_h = min(max_num_crops, num_crops_h)
        num_crops_w = 1

    crop_size_w = int(math.ceil(width / num_crops_w))
    crop_size_h = int(math.ceil(height / num_crops_h))

    if min(crop_size_w, crop_size_h) < min_crop_size:
        return 1, 1

    return num_crops_w, num_crops_h


def bbox_to_token_indices(x_min, y_min, x_max, y_max, crop_width, crop_height,
                          resized_size=896, tokens_per_side=16):
    width_ratio = resized_size / crop_width
    height_ratio = resized_size / crop_height
    patch_size = resized_size / tokens_per_side

    x_min_token = max(0, min(tokens_per_side - 1, int((x_min * width_ratio) // patch_size)))
    x_max_token = max(0, min(tokens_per_side - 1, int(math.ceil((x_max * width_ratio) / patch_size))))
    y_min_token = max(0, min(tokens_per_side - 1, int((y_min * height_ratio) // patch_size)))
    y_max_token = max(0, min(tokens_per_side - 1, int(math.ceil((y_max * height_ratio) / patch_size))))

    return [y * tokens_per_side + x for y in range(y_min_token, max(y_min_token + 1, y_max_token))
            for x in range(x_min_token, max(x_min_token + 1, x_max_token))]


def find_gemma3_image_tokens(image_starts, image_resolution, bbox, crop_layout):
    original_width, original_height = image_resolution
    x_min, y_min, x_max, y_max = bbox
    num_crops_w, num_crops_h = crop_layout

    token_indices = set()

    global_indices = bbox_to_token_indices(x_min, y_min, x_max, y_max, original_width, original_height)
    token_indices.update([i + image_starts[0] for i in global_indices])

    if num_crops_w > 1 or num_crops_h > 1:
        crop_width = math.ceil(original_width / num_crops_w)
        crop_height = math.ceil(original_height / num_crops_h)

        crop_idx = 1
        for h in range(num_crops_h):
            for w in range(num_crops_w):
                c_x_min = w * crop_width
                c_y_min = h * crop_height
                c_x_max = c_x_min + crop_width
                c_y_max = c_y_min + crop_height

                inter_x_min = max(x_min, c_x_min)
                inter_y_min = max(y_min, c_y_min)
                inter_x_max = min(x_max, c_x_max)
                inter_y_max = min(y_max, c_y_max)

                if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                    rel_x_min = inter_x_min - c_x_min
                    rel_y_min = inter_y_min - c_y_min
                    rel_x_max = inter_x_max - c_x_min
                    rel_y_max = inter_y_max - c_y_min

                    local_indices = bbox_to_token_indices(
                        rel_x_min, rel_y_min, rel_x_max, rel_y_max,
                        crop_width, crop_height
                    )
                    token_indices.update([i + image_starts[crop_idx] for i in local_indices])

                crop_idx += 1

    return list(token_indices)


def box_to_bbox(box):
    return box[0][0], box[0][1], box[1][0], box[1][1]


def max_sim(ref_features, tgt_features):
    """ColBERT MaxSim: mean of per-ref-token maximum cosine similarity."""
    ref_norm = torch.nn.functional.normalize(ref_features, p=2, dim=1)
    tgt_norm = torch.nn.functional.normalize(tgt_features, p=2, dim=1)
    sim_matrix = torch.matmul(ref_norm, tgt_norm.T)
    return torch.mean(torch.max(sim_matrix, dim=1).values).item()




# -- Main evaluation loop ------------------------------------------------------

correct_per_layer = [0] * N_LAYERS
n_dataset = len(test_dataset)

for k in tqdm(range(n_dataset)):
    image1 = Image.open("SPair-71k/" + test_dataset[k]["og_src_img_path"]).convert("RGB")
    image2 = Image.open("SPair-71k/" + test_dataset[k]["og_tgt_img_path"]).convert("RGB")

    image1_orig_size = image1.size
    image2_orig_size = image2.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image1},
                {"type": "image", "image": image2},
                {"type": "text", "text": test_dataset[k]["prompt"]},
            ],
        }
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor([image1, image2], text_input, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs.to(model.device), output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

    num_crops_1 = get_pan_and_scan_layout(image1_orig_size[0], image1_orig_size[1], processor)
    num_crops_1_total = 1 + (num_crops_1[0] * num_crops_1[1] if num_crops_1 != (1, 1) else 0)

    num_crops_2 = get_pan_and_scan_layout(image2_orig_size[0], image2_orig_size[1], processor)
    num_crops_2_total = 1 + (num_crops_2[0] * num_crops_2[1] if num_crops_2 != (1, 1) else 0)

    all_image_starts = [i for i, x in enumerate(inputs.input_ids[0].tolist()) if x == VISION_START_ID]

    if len(all_image_starts) != num_crops_1_total + num_crops_2_total:
        print(f"WARNING: Expected {num_crops_1_total + num_crops_2_total} crops, found {len(all_image_starts)}")
        print(f"image1 size: {image1_orig_size}, num_crops_1: {num_crops_1}")
        print(f"image2 size: {image2_orig_size}, num_crops_2: {num_crops_2}")
        num_crops_1 = (1, 1)
        num_crops_1_total = 1
        num_crops_2 = (1, 1)
        num_crops_2_total = 1

    image1_starts = all_image_starts[0:num_crops_1_total]
    image2_starts = all_image_starts[num_crops_1_total:num_crops_1_total + num_crops_2_total]

    ref_idx = find_gemma3_image_tokens(
        image1_starts, image1_orig_size, box_to_bbox(test_dataset[k]["ref_box"]), num_crops_1
    )
    tgt_idx = find_gemma3_image_tokens(
        image2_starts, image2_orig_size, box_to_bbox(test_dataset[k]["tgt_box"]), num_crops_2
    )
    other_idx_list = [
        find_gemma3_image_tokens(image2_starts, image2_orig_size, box_to_bbox(ob), num_crops_2)
        for ob in test_dataset[k]["other_boxes"]
    ]

    for layer in range(N_LAYERS):
        hs = all_hidden_states[layer]
        ref_rep = hs[0, ref_idx, :]
        tgt_rep = hs[0, tgt_idx, :]
        other_reps = [hs[0, ob_idx, :] for ob_idx in other_idx_list]

        tgt_score = max_sim(ref_rep, tgt_rep)
        other_scores = [max_sim(ref_rep, o) for o in other_reps]
        if tgt_score > max(other_scores):
            correct_per_layer[layer] += 1

    torch.cuda.empty_cache()


# -- Results -------------------------------------------------------------------

print("\nResults per layer:")
for layer in range(N_LAYERS):
    acc = correct_per_layer[layer] / n_dataset
    print(f"Layer {layer}: {acc:.4f} ({correct_per_layer[layer]}/{n_dataset})")

# -- Plot ----------------------------------------------------------------------

accuracies = [correct_per_layer[l] / n_dataset for l in range(N_LAYERS)]

plt.figure()
plt.plot(range(N_LAYERS), accuracies)

max_acc = max(accuracies)
max_layer = accuracies.index(max_acc)
plt.axhline(y=max_acc, alpha=0.5, color='red', linestyle='--',
            label=f'Max Accuracy: {max_acc:.4f} at Layer {max_layer}')

plt.legend()
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.xlim(0, N_LAYERS)
plt.ylim(0, 1)

with open(OUTPUT_FILE + ".pkl", "wb") as f:
    pickle.dump(correct_per_layer, f)

plt.savefig(OUTPUT_FILE + ".png")

