import argparse
import json
import os
from collections import defaultdict
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument("--direct", type=lambda x: x.lower() == 'true', required=True)
parser.add_argument("--batch", type=int, required=True)
parser.add_argument("--cuda", type=int, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

import torch
from torch.utils.data import DataLoader
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
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
BATCH_SIZE = args.batch

print(f"Using model: {args.model_path}")

OUTPUT_FILE = (
    f"squiggles_results/QWEN/direct{str(args.direct)}_dataset{args.dataset}_"
    f"model{args.model_path.replace('/', '_')}.jsonl"
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def collate_fn(batch, processor):
    messages_list = []
    answers_list = []
    shapes_list = []

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
                {"type": "text", "text": item["prompt"] + COT_USER_PROMPT_SUFFIX},
            ]}]

        messages_list.append(conversation)
        answers_list.append(item["answer"])
        shapes_list.append(item["shape_types"])

    image_inputs, video_inputs = process_vision_info(messages_list)
    texts = processor.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)

    if GET_IMMEDIATE_RESPONSE:
        texts = [x + DIRECT_ANSWER_STARTER for x in texts]

    inputs = processor(
        text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    inputs["ground_truth_answers"] = answers_list
    inputs["shape_types"] = shapes_list

    return inputs


def main():
    test_dataset = VisionLanguageDataset(dataset=args.dataset)
    print(f"Test: {len(test_dataset)}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).cuda()

    extractor_processor, extractor_model = (None, None)
    if not GET_IMMEDIATE_RESPONSE:
        extractor_processor, extractor_model = load_extractor()

    _collate_fn = partial(collate_fn, processor=processor)
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate_fn)

    model.eval()
    correct, total = 0, 0
    shape_correctness = defaultdict(int)
    shape_total = defaultdict(int)
    correctness_results = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            answers = batch.pop('ground_truth_answers')
            shapes = batch.pop('shape_types')
            batch = {k: v.cuda() for k, v in batch.items()}

            max_new_tokens = 5 if GET_IMMEDIATE_RESPONSE else 1024
            outputs = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
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
            for long_response, short_response, answer, shape in zip(
                responses, short_responses, answers, shapes
            ):
                ref_shape = shape[ord(answer.strip()) - ord('A')]
                correct_boolean = short_response == answer.strip()
                if correct_boolean:
                    batch_correct += 1
                    shape_correctness[ref_shape] += 1
                    correct += 1
                total += 1
                shape_total[ref_shape] += 1
                correctness_results.append({
                    "long_form_response": long_response,
                    "short_form_response": short_response,
                    "answer": answer,
                    "correct": correct_boolean,
                })

            print(
                f"Batch accuracy: {batch_correct}/{len(answers)} = "
                f"{100 * batch_correct / len(answers):.2f}%"
            )

    print(
        f"\nDirect?: {args.direct} Model: {args.model_path} "
        f"Accuracy: {correct}/{total} = {100 * correct / total:.2f}%\n"
    )

    for shape, shape_correct in shape_correctness.items():
        shape_count = shape_total[shape]
        print(f"Shape {shape} accuracy: {shape_correct}/{shape_count} = {100 * shape_correct / shape_count:.2f}%")

    with open(OUTPUT_FILE, "w") as f:
        for item in correctness_results:
            f.write(json.dumps(item) + "\n")

    print("Results saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
