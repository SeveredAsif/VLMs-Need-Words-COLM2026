
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--filters", nargs="+", default=None)
parser.add_argument("--direct", type=lambda x: x.lower() == 'true')
parser.add_argument("--model_path", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--cuda", type=int)
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
from torch.utils.data import DataLoader, Subset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
import random
from functools import partial
from qwen_vl_utils import process_vision_info
import json
from data_class import VisionLanguageDataset
from extractor_util import (
    COT_USER_PROMPT_SUFFIX,
    DIRECT_ANSWER_STARTER,
    extract_answer,
    load_extractor,
    parse_prefilled_answer,
    parse_qwen_response,
)


GET_IMMEDIATE_RESPONSE = args.direct
BATCH_SIZE = args.batch_size
BASE_MODEL_PATH = args.model_path
CHECKPOINT_PATH = BASE_MODEL_PATH
SEED = 42
SUBSAMPLE_TEST_SIZE = 1000
OUTPUT_FILE = f"spair71k_results/QWEN/{args.filters}_direct{str(args.direct)}_model{args.model_path.split('/')[-1]}.jsonl"
TEST_DATASET = "test"
FILTER_CONDITIONS = args.filters


def collate_fn(batch, processor):
    messages_list = []
    answers_list = []
    for item in batch:
        prompt = item["prompt"]
        if not GET_IMMEDIATE_RESPONSE:
            prompt = prompt + " " + COT_USER_PROMPT_SUFFIX

        conversation = [{"role": "user", "content": [
            {"type": "image", "image": item["ref_image"]},
            {"type": "image", "image": item["tgt_image"]},
            {"type": "text", "text": prompt},
        ]}]

        messages_list.append(conversation)
        answers_list.append(item["answer"])

    image_inputs, video_inputs = process_vision_info(messages_list)
    texts = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
    if GET_IMMEDIATE_RESPONSE:
        texts = [x + DIRECT_ANSWER_STARTER for x in texts]
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs["ground_truth_answers"] = answers_list

    return inputs


if __name__ == "__main__":

    test_dataset = VisionLanguageDataset(split=TEST_DATASET, filter_conditions=FILTER_CONDITIONS)

    if SUBSAMPLE_TEST_SIZE is not None:
        random.seed(SEED)
        num_samples = min(SUBSAMPLE_TEST_SIZE, len(test_dataset))
        indices = random.sample(range(len(test_dataset)), num_samples)
        test_dataset = Subset(test_dataset, indices)

    print(f"Test: {len(test_dataset)}")

    # Load model and processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, padding_side='left')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CHECKPOINT_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
    ).cuda()

    if not GET_IMMEDIATE_RESPONSE:
        extractor_processor, extractor_model = load_extractor()

    # Create dataloader
    _collate_fn = partial(collate_fn, processor=processor)
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_fn)

    # Evaluate
    model.eval()
    correct, total = 0, 0

    correctness_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            answers = batch.pop('ground_truth_answers')
            batch = {k: v.cuda() for k, v in batch.items()}

            if GET_IMMEDIATE_RESPONSE:
                MAX_NEW_TOKENS = 5
            else:
                MAX_NEW_TOKENS = 1024
            outputs = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            preds = processor.batch_decode(outputs, skip_special_tokens=False)

            responses = [parse_qwen_response(pred) for pred in preds]

            if GET_IMMEDIATE_RESPONSE:
                short_responses = [
                    parse_prefilled_answer(response, DIRECT_ANSWER_STARTER)
                    for response in responses
                ]
            else:
                short_responses = extract_answer(responses, extractor_processor, extractor_model)

            batch_correct = 0
            for long_response, short_response, answer in zip(responses, short_responses, answers):

                correct_boolean = answer.strip() in short_response or short_response.strip() in answer.strip()
                if correct_boolean:
                    batch_correct += 1
                    correct += 1
                total += 1
                correctness_results.append({
                    "long_form_response": long_response,
                    "short_form_response": short_response,
                    "answer": answer,
                    "correct": correct_boolean
                })

    print(f"\nFilter Conditions: {str(FILTER_CONDITIONS)} Direct?: {args.direct} Model Path: {args.model_path} Accuracy: {correct}/{total} = {100*correct/total:.2f}%\n")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for item in correctness_results:
            f.write(json.dumps(item) + "\n")

    print("Results saved to", OUTPUT_FILE)
