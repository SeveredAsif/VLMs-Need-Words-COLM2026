"""
Full fine-tuning of Gemma3-4B-IT on squiggle naming dataset.
Augmentation is done on-the-fly in the dataset — no pre-generation needed.
Uses accelerate for multi-GPU, bf16, cosine LR with warmup.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name_def_list", type=str)
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformers import (
    Gemma3ForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed

from utils import (
    NameConfig,
    build_entries,
    build_sample_image,
    configure_names,
    evaluate,
    evaluate_generation,
)

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-3-4b-it"
SRC_DIR = Path("squiggles_30")
OUTPUT_DIR = Path(args.name_def_list + "_gemma3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_EPOCHS = 1
LR = 1e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LEN = 4096
LOG_EVERY = 25
ACC_THRESHOLD = 0.90
FREEZE_VISION_AND_PROJECTOR = False
EVAL_GEN_SAMPLES = 250
SAMPLES_PER_SQUIGGLE = 2500
VAL_FRACTION = 0.2

TRAIN_TASKS = {"identify", "yesno", "mc", "ab", "describe"}
EVAL_TASKS = {"identify", "yesno", "mc", "ab", "ref", "abcd"}

NAME_CONFIG: NameConfig = configure_names(args.name_def_list)


# ── Dataset ─────────────────────────────────────────────────────────────────

class SquiggleDataset(Dataset):
    def __init__(self, entries, src_images, processor, augment=True, tasks=None):
        self.entries = entries
        self.src_images = src_images
        self.processor = processor
        self.augment = augment
        self.tasks = tasks or TRAIN_TASKS

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        squiggle_id, name = self.entries[idx]
        image, question, answer, task = build_sample_image(
            squiggle_id, name, self.src_images, self.augment, self.tasks, NAME_CONFIG,
        )

        content = [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text], images=[image],
            padding=False, truncation=True, max_length=MAX_SEQ_LEN,
            return_tensors="pt",
        )

        squeezed = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                squeezed[k] = v.squeeze(0)
            elif isinstance(v, list) and len(v) == 1 and isinstance(v[0], torch.Tensor):
                squeezed[k] = v[0]
            else:
                squeezed[k] = v
        inputs = squeezed

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        assistant_marker = "<start_of_turn>model"
        decoded = self.processor.decode(input_ids, skip_special_tokens=False)
        assistant_idx = decoded.find(assistant_marker)
        prefix_text = decoded[:assistant_idx + len(assistant_marker)]
        prefix_tokens = self.processor.tokenizer.encode(prefix_text, add_special_tokens=False)
        prompt_len = len(prefix_tokens)
        labels[:prompt_len] = -100

        inputs["labels"] = labels
        inputs["_answer"] = answer
        inputs["_task"] = task

        return inputs


def collate_fn(batch, processor):
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())

    padded = {}
    meta_keys = {"_answer", "_task"}

    for key in all_keys:
        if key in meta_keys:
            padded[key] = [sample[key] for sample in batch]
            continue
        values = [sample[key] for sample in batch if key in sample]
        if not values:
            continue
        if key == "pixel_values":
            padded[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], torch.Tensor):
            if key == "labels":
                padded[key] = nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=-100
                )
            elif key == "attention_mask":
                padded[key] = nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=0
                )
            else:
                padded[key] = nn.utils.rnn.pad_sequence(
                    values, batch_first=True, padding_value=processor.tokenizer.pad_token_id
                )
        else:
            padded[key] = values

    return padded


# ── Training ────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        mixed_precision="bf16",
        log_with=None,
    )

    train_entries, val_entries, src_images = build_entries(
        SRC_DIR, NAME_CONFIG.name_map, SAMPLES_PER_SQUIGGLE, VAL_FRACTION,
    )
    accelerator.print(f"Train: {len(train_entries)}, Val: {len(val_entries)}")

    accelerator.print(f"Loading {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, padding_side='left', do_pan_and_scan=False)

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if FREEZE_VISION_AND_PROJECTOR:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        if hasattr(model, "multi_modal_projector"):
            for param in model.multi_modal_projector.parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Frozen vision+projector: {trainable:,} / {total:,} params trainable")

    train_dataset = SquiggleDataset(train_entries, src_images, processor, augment=True, tasks=TRAIN_TASKS)
    val_dataset = SquiggleDataset(val_entries, src_images, processor, augment=False, tasks=EVAL_TASKS)

    collate = partial(collate_fn, processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=PER_DEVICE_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=8,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    num_training_steps = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    accelerator.print(f"Training steps: {num_training_steps}, Warmup: {num_warmup_steps}")
    accelerator.print(f"Effective batch size: {PER_DEVICE_BATCH_SIZE * accelerator.num_processes * GRAD_ACCUM_STEPS}")

    best_val_loss = float("inf")
    per_task = {}
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0

        for step, batch in enumerate(train_loader):
            model_batch = {k: v for k, v in batch.items() if not k.startswith("_")}

            with accelerator.accumulate(model):
                outputs = model(**model_batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            n_tokens = (batch["labels"] != -100).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens

            if accelerator.sync_gradients:
                global_step += 1

            if step % LOG_EVERY == 0:
                current_lr = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"  Epoch {epoch+1}/{NUM_EPOCHS} | Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | LR: {current_lr:.2e}"
                )

        avg_train_loss = epoch_loss / epoch_tokens
        accelerator.print(f"\nEpoch {epoch+1} train loss: {avg_train_loss:.4f}")

        val_loss = evaluate(model, val_loader, accelerator)
        accelerator.print(f"Epoch {epoch+1} val loss:   {val_loss:.4f}")

        if accelerator.is_main_process:
            acc, per_task = evaluate_generation(
                model, val_entries, src_images, processor, accelerator,
                EVAL_TASKS, EVAL_GEN_SAMPLES, NAME_CONFIG,
            )
            accelerator.print(f"Epoch {epoch+1} val acc:    {acc:.2%}")
            for t, a in sorted(per_task.items()):
                accelerator.print(f"  {t}: {a:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.print(f"New best val loss! Saving checkpoint...")
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                unwrapped.save_pretrained(OUTPUT_DIR / "best")
                processor.save_pretrained(OUTPUT_DIR / "best")

        if accelerator.is_main_process and per_task:
            if all(a >= ACC_THRESHOLD for a in per_task.values()):
                accelerator.print(
                    f"All tasks above {ACC_THRESHOLD:.0%} accuracy — stopping early!"
                )
                break

    accelerator.wait_for_everyone()
    accelerator.unwrap_model(model)
    accelerator.print("Training complete!")


if __name__ == "__main__":
    main()
