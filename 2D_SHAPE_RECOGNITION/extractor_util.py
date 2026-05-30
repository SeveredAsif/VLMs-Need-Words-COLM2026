import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

EXTRACTOR_MODEL_PATH = "Qwen/Qwen3-VL-8B-Instruct"

DIRECT_ANSWER_STARTER = "The correct option is ("
COT_USER_PROMPT_SUFFIX = "Think step by step before selecting an answer."
EXTRACTED_ANSWER_STARTER = "The extracted answer is ("
VALID_ANSWERS = {"A", "B", "C", "D"}

EXTRACTION_TEMPLATE = """You are given the long form response to a multiple choice question. Your task is to extract the answer from the response. Respond with only one letter: A, B, C, or D. Do not include any other text such as "Answer: ", "The answer is: " or parenthesis in your response. Respond with only the single letter, no other text. Do NOT think, just extract the answer.

Response: {response}"""


def parse_prefilled_answer(response, starter):
    answer = response[len(starter):].strip()[0].upper()
    if answer not in VALID_ANSWERS:
        raise ValueError(f"Expected A, B, C, or D after {starter!r}, got {response!r}")
    return answer


def parse_gemma_response(pred):
    return pred.split("<start_of_turn>model\n")[-1].split("<end_of_turn>")[0].strip()


def parse_qwen_response(pred):
    return pred.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()


def parse_intern_response(pred):
    return pred.split("\nassistant\n")[-1].strip()


def load_extractor():
    processor = AutoProcessor.from_pretrained(
        EXTRACTOR_MODEL_PATH, trust_remote_code=True, padding_side="left"
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        EXTRACTOR_MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).cuda()
    model.eval()
    return processor, model


def extract_answer(long_form_responses, processor, model):
    all_messages = []
    for long_form_response in long_form_responses:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": EXTRACTION_TEMPLATE.format(response=long_form_response),
                    }
                ],
            }
        ]
        all_messages.append(messages)

    messages_text = processor.apply_chat_template(
        all_messages, tokenize=False, add_generation_prompt=True
    )
    messages_text = [x + EXTRACTED_ANSWER_STARTER for x in messages_text]
    inputs = processor(text=messages_text, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    responses = processor.batch_decode(outputs, skip_special_tokens=False)

    short_responses = []
    for response in responses:
        short_response = parse_qwen_response(response)
        short_responses.append(parse_prefilled_answer(short_response, EXTRACTED_ANSWER_STARTER))

    return short_responses
