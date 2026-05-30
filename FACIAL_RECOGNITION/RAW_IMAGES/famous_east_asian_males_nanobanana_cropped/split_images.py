from pathlib import Path
from PIL import Image

input_dir = Path(__file__).parent

for png in input_dir.glob("*.png"):
    img = Image.open(png)
    w, h = img.size
    hw, hh = w // 2, h // 2

    out_dir = input_dir / png.stem
    out_dir.mkdir(exist_ok=True)

    positions = [(0, 0), (hw, 0), (0, hh), (hw, hh)]
    for i, (x, y) in enumerate(positions, start=1):
        crop = img.crop((x, y, x + hw, y + hh))
        crop.save(out_dir / f"{i}.png")
        print(f"Saved {out_dir.name}/{i}.png")

print("Done!")
