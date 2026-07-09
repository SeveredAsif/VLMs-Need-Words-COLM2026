import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--msize", type=str)
parser.add_argument("--direct", type=lambda x: x.lower() == 'true')
parser.add_argument("--batch", type=int)
parser.add_argument("--cuda", type=int)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
import json
import random
from functools import partial
from torch.utils.data import DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset
from extractor_util import extract_answer, load_extractor
from qwen_vl_utils import process_vision_info

DIRECT_PREFIX = "The correct option is ("
THOUGHT_SUFFIX = "Think step by step before selecting an option."
GET_IMMEDIATE_RESPONSE = args.direct
BATCH_SIZE = args.batch
BASE_MODEL_PATH = f"Qwen/Qwen3-VL-{args.msize}-Instruct"
CHECKPOINT_PATH = "../FINAL_SQUIGGLE_EXPERIMENT/qwen_2b_finetuned_on_task/best_model"
# CHECKPOINT_PATH = BASE_MODEL_PATH
DATASET_NAME = args.dataset_name
SPLIT = "test"
SEED = 42
OUTPUT_FILE = f"faces_results/QWEN_trained/direct{args.direct}_{args.msize}_{DATASET_NAME}_{CHECKPOINT_PATH}.jsonl"

SUBSAMPLE_TEST_SIZE = None
random.seed(SEED)

print(f"Evaluating Model: {CHECKPOINT_PATH}, Dataset: {DATASET_NAME}")


def collate_fn(batch, processor):
    messages_list = []
    answers_list = []
    refer_persons_list = []
    for item in batch:
        if GET_IMMEDIATE_RESPONSE:
            conversation = [{"role": "user", "content": [
                {"type": "image", "image": item["ref_image"]},
                {"type": "image", "image": item["tgt_image"]},
                {"type": "text", "text": item["prompt"]},
            ]}]
        else:
            conversation = [{"role": "user", "content": [
                {"type": "image", "image": item["ref_image"]},
                {"type": "image", "image": item["tgt_image"]},
                {"type": "text", "text": item["prompt"]+" "+THOUGHT_SUFFIX},
            ]}]
        messages_list.append(conversation)
        answers_list.append(item["answer"])
        refer_persons_list.append(item["refer_person"])
    image_inputs, video_inputs = process_vision_info(messages_list)
    texts = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
    if GET_IMMEDIATE_RESPONSE:
        texts = [x + DIRECT_PREFIX for x in texts]
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs["ground_truth_answers"] = answers_list
    inputs["refer_person"] = refer_persons_list
    return inputs


if __name__ == "__main__":
    test_dataset = VisionLanguageDataset(dataset_name=DATASET_NAME)

    if SUBSAMPLE_TEST_SIZE is not None:
        test_dataset = test_dataset.sample(SUBSAMPLE_TEST_SIZE, seed=SEED)

    print(f"Test: {len(test_dataset)}")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, padding_side='left')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CHECKPOINT_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
    ).cuda()
    extraction_processor, extraction_model = load_extractor()

    _collate_fn = partial(collate_fn, processor=processor)
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_fn)

    model.eval()
    correct, total = 0, 0
    correctness_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            answers = batch.pop('ground_truth_answers')
            refer_persons = batch.pop('refer_person')
            batch = {k: v.cuda() for k, v in batch.items()}
            MAX_NEW_TOKENS = 5 if GET_IMMEDIATE_RESPONSE else 1024
            outputs = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            preds = processor.batch_decode(outputs, skip_special_tokens=False)

            responses = []
            for pred in preds:
                response = pred.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
                responses.append(response)

            if GET_IMMEDIATE_RESPONSE:
                starter_length = len(DIRECT_PREFIX)
                short_responses = [r[starter_length:][0] for r in responses]
            else:
                short_responses = extract_answer(responses, extraction_processor, extraction_model)

            for long_response, short_response, answer, refer_person in zip(responses, short_responses, answers, refer_persons):
                correct_boolean = answer.strip() in short_response or short_response.strip() in answer.strip()
                if correct_boolean:
                    correct += 1
                total += 1
                correctness_results.append({
                    "long_form_response": long_response,
                    "short_form_response": short_response,
                    "answer": answer,
                    "refer_person": refer_person,
                    "correct": correct_boolean
                })

    print(f"\nDataset: {DATASET_NAME} Direct: {args.direct} Accuracy: {correct}/{total} = {100*correct/total:.2f}%\n")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for item in correctness_results:
            f.write(json.dumps(item) + "\n")
    print("Results saved to", OUTPUT_FILE)
