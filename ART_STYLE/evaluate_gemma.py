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
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from data_class import VisionLanguageDataset


GET_IMMEDIATE_RESPONSE = args.direct
BATCH_SIZE = args.batch
BASE_MODEL_PATH = f"google/gemma-3-{args.msize}-it"
CHECKPOINT_PATH = BASE_MODEL_PATH
# CHECKPOINT_PATH = "../FINAL_SQUIGGLE_EXPERIMENT/gemma_3_4b_it_finetuned_on_task/best_model"
DATASET_NAME = args.dataset_name
SEED = 42
OUTPUT_FILE = f"results/GEMMA/direct{args.direct}_{args.msize}_{DATASET_NAME.split("/")[0]}.jsonl"

SUBSAMPLE_TEST_SIZE = None
random.seed(SEED)

print(f"Evaluating Model: {CHECKPOINT_PATH}, Dataset: {DATASET_NAME}")


EXTRACTION_TEMPLATE = """You are given the long form response to a multiple choice question. Your task is to extract the answer from the response. Respond with only one letter: A, B, C, or D. Do not include any other text such as "Answer: ", "The answer is: " or parenthesis in your response. Respond with only the single letter, no other text. Do NOT think, just extract the answer.

Response: {response}"""

def extract_answer(long_form_responses, processor, model):
    all_messages = []
    for long_form_response in long_form_responses:
        messages = [{"role": "user", "content": [{"type": "text", "text": EXTRACTION_TEMPLATE.format(response=long_form_response)}]}]
        all_messages.append(messages)
    messages_text = processor.apply_chat_template(all_messages, tokenize=False, add_generation_prompt=True)
    messages_text = [x + "The extracted answer is (" for x in messages_text]
    inputs = processor(text=messages_text, return_tensors="pt", padding=True).to(model.device, dtype=torch.bfloat16)
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    responses = processor.batch_decode(outputs, skip_special_tokens=False)
    short_responses = []
    for response in responses:
        short_response = response.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
        short_response = short_response[len("The extracted answer is ("):][0].strip()
        short_responses.append(short_response)
    return short_responses


def collate_fn(batch, processor):
    messages_list = []
    answers_list = []
    images_list = []
    refer_names_list = []
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
                {"type": "text", "text": item["prompt"]+" Think step by step before selecting an option."},
            ]}]
        images_list.append([item["ref_image"], item["tgt_image"]])
        messages_list.append(conversation)
        answers_list.append(item["answer"])
        refer_names_list.append(item["refer_name"])
    texts = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
    if GET_IMMEDIATE_RESPONSE:
        texts = [x + "The correct answer is (" for x in texts]
    inputs = processor(images_list, texts, padding=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    inputs["ground_truth_answers"] = answers_list
    inputs["refer_name"] = refer_names_list
    return inputs


if __name__ == "__main__":
    test_dataset = VisionLanguageDataset(dataset_name=DATASET_NAME)

    if SUBSAMPLE_TEST_SIZE is not None:
        test_dataset = test_dataset.sample(SUBSAMPLE_TEST_SIZE, seed=SEED)

    print(f"Test: {len(test_dataset)}")

    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, padding_side='left', do_pan_and_scan=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(CHECKPOINT_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True).cuda()
    model.eval()

    extraction_model = Gemma3ForConditionalGeneration.from_pretrained(BASE_MODEL_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True).cuda()
    extraction_model.eval()

    _collate_fn = partial(collate_fn, processor=processor)
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_fn)

    model.eval()
    correct, total = 0, 0
    correctness_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            answers = batch.pop('ground_truth_answers')
            refer_names = batch.pop('refer_name')
            batch = {k: v.cuda() for k, v in batch.items()}
            MAX_NEW_TOKENS = 5 if GET_IMMEDIATE_RESPONSE else 1024
            outputs = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            preds = processor.batch_decode(outputs, skip_special_tokens=False)

            responses = []
            for pred in preds:
                response = pred.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()
                responses.append(response)

            if GET_IMMEDIATE_RESPONSE:
                starter_length = len("The correct answer is (")
                short_responses = [r[starter_length:][0] for r in responses]
            else:
                short_responses = extract_answer(responses, processor, extraction_model)

            for long_response, short_response, answer, refer_name in zip(responses, short_responses, answers, refer_names):
                correct_boolean = answer.strip() in short_response or short_response.strip() in answer.strip()
                if correct_boolean:
                    correct += 1
                total += 1
                correctness_results.append({
                    "long_form_response": long_response,
                    "short_form_response": short_response,
                    "answer": answer,
                    "refer_name": refer_name,
                    "correct": correct_boolean
                })

    print(f"\nDataset: {DATASET_NAME} Direct: {args.direct} Accuracy: {correct}/{total} = {100*correct/total:.2f}%\n")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for item in correctness_results:
            f.write(json.dumps(item) + "\n")
    print("Results saved to", OUTPUT_FILE)
