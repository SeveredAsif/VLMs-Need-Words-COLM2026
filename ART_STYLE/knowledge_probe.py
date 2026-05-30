#!/usr/bin/env python3
"""Test whether a vision-language model can identify painters and paintings."""

import argparse
import csv
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from PIL import Image
from transformers import AutoProcessor

PAINTINGS_ROOT = Path(__file__).resolve().parent / "paintings"
MANIFEST = PAINTINGS_ROOT / "image_sources.csv"

PAINTER_PROMPT = "Which painter created this painting?"
PAINTING_PROMPT = "What is the name of the painting?"
JUDGE_PROMPT = (
    "Prediction: {prediction}\n"
    "Ground truth: {ground_truth}\n\n"
    "Does the prediction match the ground truth? Answer Yes or No only."
)


def model_family(model_name):
    lower = model_name.lower()
    if "gemma" in lower:
        return "gemma"
    if "qwen" in lower:
        return "qwen"
    raise ValueError(f"Unsupported model (expected Qwen3 or Gemma3): {model_name}")


def default_output_path(model_name):
    slug = model_name.replace("/", "_")
    return Path(__file__).resolve().parent / f"{slug}_painting_identification_results.csv"


def load_model(model_name):
    family = model_family(model_name)
    if family == "gemma":
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).cuda()
        return model

    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).cuda()
    return model


def load_paintings(manifest_path):
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    samples = []
    for row in rows:
        image_path = PAINTINGS_ROOT / row["filename"]
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue
        samples.append(
            {
                "filename": row["filename"],
                "image_path": image_path,
                "painter_gt": row["artist"],
                "painting_gt": row["painting_title"],
                "bucket": row["bucket"],
                "painter_dir": row["painter"],
            }
        )
    return samples


def decode_response(processor, outputs, model_family):
    response = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    if model_family == "gemma":
        return response.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
    return response.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()


def query_model(model, processor, image, prompt, model_family):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"} if model_family == "gemma" else {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if model_family == "gemma":
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    return decode_response(processor, outputs, model_family)


def query_text_model(model, processor, prompt, model_family):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    return decode_response(processor, outputs, model_family)


def parse_judge_answer(response):
    normalized = response.strip().lower()
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    return False


def judge_match(model, processor, prediction, ground_truth, model_family):
    prompt = JUDGE_PROMPT.format(prediction=prediction, ground_truth=ground_truth)
    judge_response = query_text_model(model, processor, prompt, model_family)
    return judge_response, parse_judge_answer(judge_response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model id, e.g. Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    family = model_family(args.model)
    output_path = args.output or default_output_path(args.model)

    samples = load_paintings(args.manifest)
    if not samples:
        raise SystemExit(f"No paintings found in {args.manifest}")

    print(f"\nLoading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    model = load_model(args.model)
    model.eval()

    results = []
    painter_correct = painting_correct = 0

    for sample in samples:
        print(f"\n{'=' * 60}")
        print(f"Testing: {sample['painting_gt']} ({sample['painter_gt']})")

        image = Image.open(sample["image_path"]).convert("RGB")

        painter_pred = query_model(model, processor, image, PAINTER_PROMPT, family)
        painting_pred = query_model(model, processor, image, PAINTING_PROMPT, family)

        painter_judge_response, painter_match = judge_match(
            model, processor, painter_pred, sample["painter_gt"], family
        )
        painting_judge_response, painting_match = judge_match(
            model, processor, painting_pred, sample["painting_gt"], family
        )

        painter_correct += int(painter_match)
        painting_correct += int(painting_match)

        print(f"Painter prediction: {painter_pred}")
        print(f"Painter judge: {painter_judge_response} ({'Yes' if painter_match else 'No'})")
        print(f"Painting prediction: {painting_pred}")
        print(f"Painting judge: {painting_judge_response} ({'Yes' if painting_match else 'No'})")

        results.append(
            {
                "filename": sample["filename"],
                "painter_dir": sample["painter_dir"],
                "bucket": sample["bucket"],
                "painter_gt": sample["painter_gt"],
                "painting_gt": sample["painting_gt"],
                "painter_prediction": painter_pred,
                "painting_prediction": painting_pred,
                "painter_judge_response": painter_judge_response,
                "painting_judge_response": painting_judge_response,
                "painter_correct": "Yes" if painter_match else "No",
                "painting_correct": "Yes" if painting_match else "No",
            }
        )

    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Painter accuracy: {painter_correct}/{total} = {100 * painter_correct / total:.1f}%")
    print(f"Painting accuracy: {painting_correct}/{total} = {100 * painting_correct / total:.1f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
