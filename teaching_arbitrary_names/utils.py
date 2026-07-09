"""Shared squiggle naming dataset utilities for VLM fine-tuning."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import random
import torch
from names import human_names, ordinary_names, random_names, similar_names
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

NAMES = [
    "squiggle_0", "squiggle_1", "squiggle_2", "squiggle_3", "squiggle_4",
    "squiggle_5", "squiggle_6", "squiggle_7", "squiggle_8", "squiggle_9",
]

IDENTIFY_TEMPLATES = [
    "What is this shape called?",
    "Name this shape.",
    "What shape is shown in this image?",
    "Identify this shape.",
    "What do you see in this image?",
    "Can you tell me the name of this shape?",
    "What is the name of this shape?",
    "This is a picture of a shape. What is it?",
    "Look at this image. What shape is this?",
    "What is this?",
    "Tell me the name of this shape.",
    "What shape is depicted here?",
    "Recognize this shape and tell me its name.",
    "What's the name of the shape in this picture?",
    "Identify the shape shown.",
]

YESNO_TEMPLATES_POS = [
    "Is this a {name}?",
    "Is the shape in this image a {name}?",
    "Does this image show a {name}?",
    "Is this shape called {name}?",
    "Would you call this a {name}?",
]

YESNO_TEMPLATES_NEG = [
    "Is this a {wrong}?",
    "Is the shape in this image a {wrong}?",
    "Does this image show a {wrong}?",
    "Is this shape called {wrong}?",
    "Would you call this a {wrong}?",
]

MC_TEMPLATES = [
    "Which of the following is this shape: {choices}?",
    "Select the correct name for this shape from: {choices}.",
    "This shape is one of the following: {choices}. Which one?",
    "From the options {choices}, which is this shape?",
]

DESCRIBE_TEMPLATES = [
    "Describe what a {name} looks like.",
    "What does a {name} look like?",
    "Can you describe the shape called {name}?",
    "Tell me about the visual appearance of a {name}.",
    "How would you describe a {name}?",
]

AB_LABEL_PAIRS = [
    ("A", "B"), ("1", "2"), ("X", "Y"), ("P", "Q"),
    ("L", "R"), ("I", "II"), ("REF1", "REF2"),
]

AB_TEMPLATES = [
    "Which object is the {name}, {left} or {right}?",
    "Is the {name} {left} or {right}?",
    "Which one is the {name}, {left} or {right}?",
    "Identify which is the {name}: {left} or {right}?",
    "{left} or {right} — which is the {name}?",
]

REF_TEMPLATES = [
    "What is the name of the REF object?",
    "What is the REF shape called?",
    "Identify the shape labeled REF.",
    "Name the object marked REF.",
    "Which shape is the REF one? What is its name?",
    "The shape labeled REF is called what?",
]

ABCD_TEMPLATES = [
    "Which option is the {name}? A, B, C, or D?",
    "Which of A, B, C, D is the {name}?",
    "Identify the {name}: is it A, B, C, or D?",
    "The {name} is which option — A, B, C, or D?",
    "A, B, C, or D — which is the {name}?",
]


@dataclass
class NameConfig:
    name_def_dict: dict
    name_map: dict
    reverse_name_map: dict
    all_names: list
    descriptions: dict


def configure_names(name_def_list: str) -> NameConfig:
    if name_def_list == "random_names":
        name_def_dict = random_names
    elif name_def_list == "human_names":
        name_def_dict = human_names
    elif name_def_list == "ordinary_names":
        name_def_dict = ordinary_names
    elif name_def_list == "similar_names":
        name_def_dict = similar_names
    else:
        raise ValueError(f"Invalid name definition list: {name_def_list}")

    print("Using name definition list: ", name_def_dict.keys())

    name_map = {}
    for id_name, semantic_name in zip(NAMES, list(name_def_dict.keys())):
        name_map[id_name] = semantic_name

    reverse_name_map = {v: k for k, v in name_map.items()}
    descriptions = {
        id_name: name_def_dict[name_map[id_name]] for id_name in NAMES
    }

    return NameConfig(
        name_def_dict=name_def_dict,
        name_map=name_map,
        reverse_name_map=reverse_name_map,
        all_names=list(name_map.values()),
        descriptions=descriptions,
    )


def recolor_stroke(img, stroke_color, bg_color):
    arr = np.array(img.convert("RGB"))
    gray = np.array(img.convert("L"))
    mask = gray < 128
    result = np.full_like(arr, bg_color)
    result[mask] = stroke_color
    return Image.fromarray(result)


def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def contrasting_colors():
    while True:
        fg = random_color()
        bg = random_color()
        lum_fg = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
        lum_bg = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        if abs(lum_fg - lum_bg) > 120:
            return fg, bg


def get_bg_color(img):
    corner = img.getpixel((0, 0))
    return corner if isinstance(corner, tuple) else (corner,) * 3


def stroke_coverage(img, bg_color):
    arr = np.array(img.convert("RGB"))
    bg = np.array(bg_color)
    diff = np.abs(arr.astype(float) - bg.astype(float)).max(axis=2)
    return (diff > 30).mean()


def augment_image(img):
    img = img.copy().convert("RGB")

    if random.random() < 0.6:
        fg, bg = contrasting_colors()
        img = recolor_stroke(img, fg, bg)
    elif random.random() < 0.3:
        img = ImageOps.invert(img)

    bg = get_bg_color(img)

    gray = np.array(img.convert("L"))
    bg_lum = int(0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2])
    stroke_mask = np.abs(gray.astype(int) - bg_lum) > 30
    if stroke_mask.any():
        rows = np.any(stroke_mask, axis=1)
        cols = np.any(stroke_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 5
        rmin = max(rmin - pad, 0)
        rmax = min(rmax + pad, img.height - 1)
        cmin = max(cmin - pad, 0)
        cmax = min(cmax + pad, img.width - 1)
        img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

    scale = random.uniform(0.7, 1.4)
    new_w = max(int(img.width * scale), 64)
    new_h = max(int(img.height * scale), 64)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    padding_ratio = random.uniform(1.05, 1.5)
    canvas_w = max(int(new_w * padding_ratio), new_w + 10)
    canvas_h = max(int(new_h * padding_ratio), new_h + 10)
    canvas_w = min(max(canvas_w, 128), 512)
    canvas_h = min(max(canvas_h, 128), 512)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    max_x = max(canvas_w - new_w, 0)
    max_y = max(canvas_h - new_h, 0)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    paste_w = min(new_w, canvas_w - x)
    paste_h = min(new_h, canvas_h - y)
    canvas.paste(img.crop((0, 0, paste_w, paste_h)), (x, y))
    img = canvas

    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.flip(img)

    if random.random() < 0.3:
        zoom_factor = random.uniform(1.5, 3.0)
        w_z, h_z = img.size
        new_cw = int(w_z * zoom_factor)
        new_ch = int(h_z * zoom_factor)
        new_cw = min(max(new_cw, 128), 768)
        new_ch = min(max(new_ch, 128), 768)
        zoomed = Image.new("RGB", (new_cw, new_ch), bg)
        zx = random.randint(0, new_cw - w_z)
        zy = random.randint(0, new_ch - h_z)
        zoomed.paste(img, (zx, zy))
        img = zoomed

    if random.random() < 0.25:
        w2, h2 = img.size
        mag = random.uniform(0.02, 0.08)
        coeffs = [random.uniform(-mag, mag) for _ in range(8)]
        try:
            img = img.transform(
                (w2, h2), Image.PERSPECTIVE,
                coeffs, resample=Image.BICUBIC, fillcolor=bg,
            )
        except Exception:
            pass

    if random.random() < 0.3:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))

    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    if random.random() < 0.15:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    coverage = stroke_coverage(img, bg)
    if coverage < 0.02:
        return None

    return img


def augment_shape(img):
    img = img.copy().convert("RGB")

    if random.random() < 0.4:
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    if random.random() < 0.25:
        w, h = img.size
        mag = random.uniform(0.02, 0.08)
        coeffs = [random.uniform(-mag, mag) for _ in range(8)]
        try:
            img = img.transform(
                (w, h), Image.PERSPECTIVE, coeffs,
                resample=Image.BICUBIC, fillcolor=(255, 255, 255),
            )
        except Exception:
            pass

    return img


def make_ab_image(src_a, src_b, label_left="A", label_right="B",
                  cell_size=300, gap=20, label_height=40):
    img_a = augment_shape(src_a).resize((cell_size, cell_size), Image.BICUBIC)
    img_b = augment_shape(src_b).resize((cell_size, cell_size), Image.BICUBIC)

    total_w = cell_size * 2 + gap
    total_h = cell_size + label_height
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    canvas.paste(img_a, (0, 0))
    canvas.paste(img_b, (cell_size + gap, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label_left, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((cell_size - tw) // 2, cell_size + 4), label_left, fill=(0, 0, 0), font=font)

    bbox = draw.textbbox((0, 0), label_right, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(
        (cell_size + gap + (cell_size - tw) // 2, cell_size + 4),
        label_right, fill=(0, 0, 0), font=font,
    )

    if random.random() < 0.6:
        fg, bg_col = contrasting_colors()
        canvas = recolor_stroke(canvas, fg, bg_col)
    elif random.random() < 0.3:
        canvas = ImageOps.invert(canvas)

    if random.random() < 0.3:
        bg_col = get_bg_color(canvas)
        zoom = random.uniform(1.3, 2.0)
        new_w = int(canvas.width * zoom)
        new_h = int(canvas.height * zoom)
        zoomed = Image.new("RGB", (new_w, new_h), bg_col)
        zx = random.randint(0, new_w - canvas.width)
        zy = random.randint(0, new_h - canvas.height)
        zoomed.paste(canvas, (zx, zy))
        canvas = zoomed

    if random.random() < 0.3:
        canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.8, 1.2))
        canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.8, 1.3))

    if random.random() < 0.15:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    if random.random() < 0.15:
        arr = np.array(canvas).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        canvas = Image.fromarray(arr)

    return canvas


def augment_composite(canvas):
    if random.random() < 0.6:
        fg, bg_col = contrasting_colors()
        canvas = recolor_stroke(canvas, fg, bg_col)
    elif random.random() < 0.3:
        canvas = ImageOps.invert(canvas)

    if random.random() < 0.15:
        bg_col = get_bg_color(canvas)
        zoom = random.uniform(1.1, 1.3)
        new_w = int(canvas.width * zoom)
        new_h = int(canvas.height * zoom)
        zoomed = Image.new("RGB", (new_w, new_h), bg_col)
        zx = random.randint(0, new_w - canvas.width)
        zy = random.randint(0, new_h - canvas.height)
        zoomed.paste(canvas, (zx, zy))
        canvas = zoomed

    if random.random() < 0.4:
        canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.4:
        canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.8, 1.3))
    if random.random() < 0.3:
        canvas = ImageEnhance.Sharpness(canvas).enhance(random.uniform(0.5, 1.5))

    if random.random() < 0.15:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    if random.random() < 0.15:
        arr = np.array(canvas).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        canvas = Image.fromarray(arr)

    return canvas


def _get_font(size=26):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def scatter_four(images, labels, canvas_w=900, canvas_h=700, shape_size_range=(150, 200), label_gap=6):
    bg = (255, 255, 255)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)
    font = _get_font(26)

    sz = random.randint(shape_size_range[0], shape_size_range[1])

    half_w = canvas_w // 2
    half_h = canvas_h // 2
    cells = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]
    random.shuffle(cells)

    positions = []
    for cx, cy in cells:
        margin = 15
        max_x = cx + half_w - sz - margin
        max_y = cy + half_h - sz - label_gap - 30 - margin
        x = random.randint(cx + margin, max(cx + margin, max_x))
        y = random.randint(cy + margin, max(cy + margin, max_y))
        positions.append((x, y, sz))

    for i, (img, label) in enumerate(zip(images, labels)):
        x, y, sz_ = positions[i]
        resized = augment_shape(img).resize((sz_, sz_), Image.BICUBIC)
        canvas.paste(resized, (x, y))
        if label:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x + (sz_ - tw) // 2, y + sz_ + label_gap), label, fill=(0, 0, 0), font=font)

    return canvas


def make_ref_image(target_img, distractor_imgs, target_id_list):
    all_imgs = [target_img] + distractor_imgs
    indices = list(range(4))
    random.shuffle(indices)
    shuffled_imgs = [all_imgs[i] for i in indices]
    target_pos = indices.index(0)

    labels = ["REF" if i == target_pos else "" for i in range(4)]
    canvas = scatter_four(shuffled_imgs, labels)
    return augment_composite(canvas)


def make_abcd_image(target_img, distractor_imgs):
    abcd = ["A", "B", "C", "D"]
    all_imgs = [target_img] + distractor_imgs
    indices = list(range(4))
    random.shuffle(indices)
    shuffled_imgs = [all_imgs[i] for i in indices]
    target_pos = indices.index(0)
    answer_label = abcd[target_pos]

    canvas = scatter_four(shuffled_imgs, abcd)
    canvas = augment_composite(canvas)
    return canvas, answer_label


def make_qa(name, allowed_tasks, name_config: NameConfig):
    task = random.choice(list(allowed_tasks))

    if task == "identify":
        q = random.choice(IDENTIFY_TEMPLATES)
        return q, name, "identify"
    if task == "yesno":
        if random.random() < 0.5:
            q = random.choice(YESNO_TEMPLATES_POS).format(name=name)
            return q, "Yes.", "yesno"
        wrong = random.choice([n for n in name_config.all_names if n != name])
        q = random.choice(YESNO_TEMPLATES_NEG).format(wrong=wrong)
        return q, f"No, this is a {name}.", "yesno"
    if task == "mc":
        distractors = random.sample([n for n in name_config.all_names if n != name], 3)
        options = distractors + [name]
        random.shuffle(options)
        q = random.choice(MC_TEMPLATES).format(choices=", ".join(options))
        return q, name, "mc"
    if task == "ab":
        label_left, label_right = random.choice(AB_LABEL_PAIRS)
        q = random.choice(AB_TEMPLATES).format(name=name, left=label_left, right=label_right)
        return q, (label_left, label_right), "ab"
    if task == "ref":
        q = random.choice(REF_TEMPLATES)
        return q, name, "ref"
    if task == "abcd":
        q = random.choice(ABCD_TEMPLATES).format(name=name)
        return q, None, "abcd"
    q = random.choice(DESCRIBE_TEMPLATES).format(name=name)
    a = name_config.descriptions[name_config.reverse_name_map[name]]
    return q, a, "describe"


def build_sample_image(squiggle_id, name, src_images, augment, tasks, name_config):
    """Load source image, pick a task, and build the augmented/composite PIL image."""
    src_img = src_images[squiggle_id]
    question, answer, task = make_qa(name, tasks, name_config)

    if task == "ab":
        label_left, label_right = answer
        other_ids = [sid for sid in src_images if sid != squiggle_id]
        distractor_id = random.choice(other_ids)
        distractor_img = src_images[distractor_id]
        if random.random() < 0.5:
            image = make_ab_image(src_img, distractor_img, label_left, label_right)
            answer = label_left
        else:
            image = make_ab_image(distractor_img, src_img, label_left, label_right)
            answer = label_right
    elif task == "ref":
        other_ids = [sid for sid in src_images if sid != squiggle_id]
        dist_ids = random.sample(other_ids, 3)
        dist_imgs = [src_images[d] for d in dist_ids]
        image = make_ref_image(src_img, dist_imgs, dist_ids)
    elif task == "abcd":
        other_ids = [sid for sid in src_images if sid != squiggle_id]
        dist_ids = random.sample(other_ids, 3)
        dist_imgs = [src_images[d] for d in dist_ids]
        image, answer = make_abcd_image(src_img, dist_imgs)
    elif augment:
        image = augment_image(src_img)
        if image is None:
            fg, bg = contrasting_colors()
            image = recolor_stroke(src_img, fg, bg)
    else:
        image = src_img.copy()

    return image, question, answer, task


def build_entries(src_dir, name_map, samples_per_squiggle, val_fraction):
    src_images = {}
    all_entries = []

    for squiggle_id, name in name_map.items():
        src_path = src_dir / f"{squiggle_id}.png"
        src_images[squiggle_id] = Image.open(src_path).convert("RGB")
        for _ in range(samples_per_squiggle):
            all_entries.append((squiggle_id, name))

    random.shuffle(all_entries)
    n_val = int(len(all_entries) * val_fraction)
    val_entries = all_entries[:n_val]
    train_entries = all_entries[n_val:]

    return train_entries, val_entries, src_images


def evaluate(model, dataloader, accelerator):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            model_batch = {k: v for k, v in batch.items() if not k.startswith("_")}
            outputs = model(**model_batch)
            n_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    model.train()
    return total_loss / total_tokens


def evaluate_generation(model, val_entries, src_images, processor, accelerator,
                        eval_tasks, eval_gen_samples, name_config):
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    subset = random.sample(val_entries, min(eval_gen_samples, len(val_entries)))

    correct = 0
    total = 0
    task_correct = {}
    task_total = {}

    with torch.no_grad():
        for squiggle_id, name in subset:
            src_img = src_images[squiggle_id]
            question, answer, task = make_qa(name, eval_tasks, name_config)

            if task == "ab":
                label_left, label_right = answer
                other_ids = [sid for sid in src_images if sid != squiggle_id]
                distractor_id = random.choice(other_ids)
                distractor_img = src_images[distractor_id]
                if random.random() < 0.5:
                    eval_img = make_ab_image(src_img, distractor_img, label_left, label_right)
                    answer = label_left
                else:
                    eval_img = make_ab_image(distractor_img, src_img, label_left, label_right)
                    answer = label_right
            elif task == "ref":
                other_ids = [sid for sid in src_images if sid != squiggle_id]
                dist_ids = random.sample(other_ids, 3)
                dist_imgs = [src_images[d] for d in dist_ids]
                eval_img = make_ref_image(src_img, dist_imgs, dist_ids)
            elif task == "abcd":
                other_ids = [sid for sid in src_images if sid != squiggle_id]
                dist_ids = random.sample(other_ids, 3)
                dist_imgs = [src_images[d] for d in dist_ids]
                eval_img, answer = make_abcd_image(src_img, dist_imgs)
            else:
                eval_img = src_img

            content = [
                {"type": "image", "image": eval_img},
                {"type": "text", "text": question},
            ]
            messages = [{"role": "user", "content": content}]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[eval_img],
                return_tensors="pt",
            )
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

            output_ids = unwrapped.generate(
                **inputs, max_new_tokens=64, do_sample=False,
            )
            prompt_len = inputs["input_ids"].shape[1]
            generated = processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True
            ).strip().lower()

            gt = answer.strip().lower()
            hit = gt in generated or generated in gt

            task_correct[task] = task_correct.get(task, 0) + int(hit)
            task_total[task] = task_total.get(task, 0) + 1
            correct += int(hit)
            total += 1

    model.train()
    acc = correct / total
    per_task = {t: task_correct.get(t, 0) / task_total[t] for t in task_total}
    return acc, per_task
