import os
from traceback import print_tb
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.data import DataLoader
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from pathlib import Path
from tqdm import tqdm
from functools import partial
from data_class import VisionLanguageDataset


class Config:
    model_name = "google/gemma-3-4b-it"
    num_epochs = 4
    batch_size = 2
    learning_rate = 1e-5
    max_grad_norm = 1.0
    gradient_accumulation_steps = 4
    num_workers = 4
    prefetch_factor = 2
    output_dir = "gemma_3_4b_it_finetuned_on_task"
    device = "cuda" if torch.cuda.is_available() else "cpu"


def train_collate_fn(batch, processor):
    texts = []
    images = []
    for item in batch:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["ref_image"]},
                    {"type": "image", "image": item["tgt_image"]},
                    {"type": "text", "text": item["prompt"]},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item["answer"]},
                ]
            }
        ]
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
        images.append([item["ref_image"], item["tgt_image"]])

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    labels_list = []
    assistant_marker = "<start_of_turn>model"

    def get_prompt_length(input_ids):
        decoded = processor.decode(input_ids, skip_special_tokens=False)
        assistant_idx = decoded.find(assistant_marker)
        prefix_text = decoded[:assistant_idx + len(assistant_marker)]
        prefix_tokens = processor.tokenizer.encode(prefix_text, add_special_tokens=False)
        return len(prefix_tokens)

    for i in range(len(inputs["input_ids"])):
        prompt_length = get_prompt_length(inputs["input_ids"][i])
        label = inputs["input_ids"][i].clone()
        label[:prompt_length] = -100
        labels_list.append(label)

    inputs["labels"] = torch.stack(labels_list)

    return inputs


def test_collate_fn(batch, processor):
    texts = []
    images = []
    answers = []

    for item in batch:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["ref_image"]},
                    {"type": "image", "image": item["tgt_image"]},
                    {"type": "text", "text": item["prompt"]},
                ]
            }
        ]
        text = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
        images.append([item["ref_image"], item["tgt_image"]])
        answers.append(item["answer"])

    texts = [x+"The correct answer is (" for x in texts]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    inputs["ground_truth_answers"] = answers
    return inputs


def freeze_vision_encoder_and_projector(model):
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'language_model' in name:
            param.requires_grad = True

    return model


def evaluate(model, processor, test_dataloader, config):
    model.eval()

    correct = 0
    total = 0

    for batch in tqdm(test_dataloader, desc="Evaluating"):
        ground_truth_answers = batch.pop('ground_truth_answers')

        batch_input = {k: v.to(config.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model.generate(
                    **batch_input,
                    max_new_tokens=10,
                    do_sample=False
                )

        predictions = processor.batch_decode(outputs, skip_special_tokens=False)

        for pred, answer in zip(predictions, ground_truth_answers):
            assistant_response = pred.split("<start_of_turn>model")[-1].split("<end_of_turn>")[0].strip()
            # print(assistant_response)
            if answer.strip() in assistant_response.strip():
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"  Correct: {correct}/{total} | Accuracy: {accuracy * 100:.2f}%")

    model.train()
    return accuracy


def train(config):
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side='left',
        do_pan_and_scan=True
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to(config.device)

    model = freeze_vision_encoder_and_projector(model)

    train_dataset = VisionLanguageDataset(dataset="squiggles_30_TRAIN")
    test30_dataset = VisionLanguageDataset(dataset="squiggles_30_TEST")
    val_dataset = VisionLanguageDataset(dataset="squiggles_30_VAL")

    test100_dataset = VisionLanguageDataset(dataset="squiggles_100_TEST")
    test50_dataset = VisionLanguageDataset(dataset="squiggles_50_TEST")
    test40_dataset = VisionLanguageDataset(dataset="squiggles_40_TEST")
    test20_dataset = VisionLanguageDataset(dataset="squiggles_20_TEST")

    test_datasets = {
        "30-iod": val_dataset,
        "30-ood": test30_dataset,
        "100-iod": test100_dataset,
        "50-iod": test50_dataset,
        "40-iod": test40_dataset,
        "20-iod": test20_dataset,
    }

    train_collate_with_processor = partial(train_collate_fn, processor=processor)
    val_collate_with_processor = partial(test_collate_fn, processor=processor)

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_collate_with_processor,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.num_workers > 0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=val_collate_with_processor,
        num_workers=min(2, config.num_workers),
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.num_workers > 0
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate
    )

    Path(config.output_dir).mkdir(exist_ok=True, parents=True)

    global_step = 0
    best_val_accuracy = 0.0
    best_model_path = Path(config.output_dir) / "best_model"
    model.train()

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / config.gradient_accumulation_steps

            loss.backward()
            epoch_loss += loss.item() * config.gradient_accumulation_steps

            if (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    config.max_grad_norm
                )

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

            progress_bar.set_postfix({
                'loss': f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                'step': global_step
            })

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        accuracy = evaluate(model, processor, val_dataloader, config)
        print(f"Val accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            best_model_path.mkdir(exist_ok=True, parents=True)
            model.save_pretrained(best_model_path)
            processor.save_pretrained(best_model_path)
            print(f"  -> New best model saved (accuracy: {accuracy * 100:.2f}%)")

    print(f"\nTraining completed! Loading best model (val accuracy: {best_val_accuracy * 100:.2f}%)")

    print(f"Using model: {best_model_path}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        best_model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to(config.device)

    for dataset_name, test_dataset in test_datasets.items():
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=val_collate_with_processor,
            num_workers=min(2, config.num_workers),
            pin_memory=True,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.num_workers > 0
        )

        accuracy = evaluate(model, processor, test_dataloader, config)
        print(f"{dataset_name} accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    config = Config()
    train(config)
