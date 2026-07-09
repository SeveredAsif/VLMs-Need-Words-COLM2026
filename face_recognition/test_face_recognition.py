#!/usr/bin/env python3
"""Test whether Qwen3-VL can recognize faces of famous people."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# name: English name, alt_names: alternative names/Chinese characters to check
FAMOUS_PEOPLE = [
    # Politicians - Wikimedia works for Xi
    {"name": "Xi Jinping", "alt_names": ["习近平"], "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/32/Xi_Jinping_2019.jpg"},
    
    # Basketball - NBA CDN works
    {"name": "Yao Ming", "alt_names": ["姚明"], "image_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/2397.png"},
    {"name": "Jeremy Lin", "alt_names": ["林书豪", "Lin"], "image_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/201152.png"},
    {"name": "Rui Hachimura", "alt_names": ["八村塁", "Hachimura"], "image_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/1629060.png"},
    {"name": "Yi Jianlian", "alt_names": ["易建联"], "image_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/101139.png"},
    
    # Soccer - Premier League works
    {"name": "Son Heung-min", "alt_names": ["손흥민", "孙兴慜", "Son"], "image_url": "https://resources.premierleague.com/premierleague/photos/players/250x250/p85971.png"},
    
    # Baseball - MLB works
    {"name": "Shohei Ohtani", "alt_names": ["大谷翔平", "Ohtani"], "image_url": "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/660271/headshot/67/current"},
    {"name": "Ichiro Suzuki", "alt_names": ["イチロー", "鈴木一朗", "Ichiro"], "image_url": "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/400085/headshot/67/current"},
    {"name": "Yu Darvish", "alt_names": ["ダルビッシュ有", "Darvish"], "image_url": "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/506433/headshot/67/current"},
    {"name": "Hyun Jin Ryu", "alt_names": ["류현진", "柳賢振", "Ryu"], "image_url": "https://img.mlbstatic.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/547943/headshot/67/current"},
    
    # Golf - PGA Tour cloudinary
    {"name": "Hideki Matsuyama", "alt_names": ["松山英樹", "Matsuyama"], "image_url": "https://pga-tour-res.cloudinary.com/image/upload/c_fill,dpr_2.0,f_auto,g_face:center,h_350,q_auto,w_280/headshots_30925.png"},
    {"name": "Tom Kim", "alt_names": ["김주형", "Kim Joo-hyung"], "image_url": "https://pga-tour-res.cloudinary.com/image/upload/c_fill,dpr_2.0,f_auto,g_face:center,h_350,q_auto,w_280/headshots_55182.png"},
    
    # Tennis - ATP
    {"name": "Kei Nishikori", "alt_names": ["錦織圭", "Nishikori"], "image_url": "https://www.atptour.com/-/media/alias/player-headshot/N552"},
]

IDENTIFICATION_PROMPT = "Who is the person in this image? Give the name only."


def download_image(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://en.wikipedia.org/',
    }
    response = requests.get(url, headers=headers, timeout=30)
    if 'image' not in response.headers.get('Content-Type', ''):
        print(f"  Warning: Got {response.headers.get('Content-Type')} instead of image, skipping...")
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def query_model(model, processor, image, prompt):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    response = processor.batch_decode(outputs, skip_special_tokens=False)[0]
    return response.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()


if __name__ == "__main__":
    print(f"\nLoading {MODEL_NAME}...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, attn_implementation="flash_attention_2", trust_remote_code=True
    ).cuda()
    model.eval()
    
    correct, total = 0, 0
    for person in FAMOUS_PEOPLE:
        print(f"\n{'='*60}")
        print(f"Testing: {person['name']}")
        
        image = download_image(person['image_url'])
        if image is None:
            continue
        
        total += 1
        response = query_model(model, processor, image, IDENTIFICATION_PROMPT)
        
        print(f"Response: {response}")
        
        # Check if name or alt names appear in response
        response_lower = response.lower()
        names_to_check = [person['name'].lower()] + person.get('alt_names', [])
        # Skip short ASCII names (< 3 chars) but keep CJK names
        names_to_check = [n for n in names_to_check if len(n) >= 3 or not n.isascii()]
        match = any(n.lower() in response_lower or n in response for n in names_to_check)
        if match:
            correct += 1
            print("✓ Correct")
        else:
            print("✗ Incorrect")
    
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")
