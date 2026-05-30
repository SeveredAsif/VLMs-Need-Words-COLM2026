# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from PIL import Image
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument('--input_folder', type=str, required=True)
parser.add_argument('--num_tasks', type=int, required=True)
parser.add_argument('--use_random_colors', type=lambda x: x.lower() in ['true', '1', 'yes'], default=True)
parser.add_argument('--dpi', type=int, default=100)
args = parser.parse_args()

def change_colors(image, color1, color2, preserve_red=False):
    """
    Interpolates a grayscale image between two RGB colors.
    
    :param image: PIL Image (mode 'L' or 'RGB')
    :param color1: Tuple (R, G, B) for the 'black' parts
    :param color2: Tuple (R, G, B) for the 'white' parts
    :param preserve_red: If True, keep red pixels (like text labels) unchanged
    :return: PIL Image in RGB mode
    """
    # Convert to RGB if not already
    rgb_image = image.convert('RGB')
    rgb_array = np.array(rgb_image).astype(float)
    
    # If preserving red, create a mask for red pixels
    if preserve_red:
        # Detect red pixels (R > 200, G < 100, B < 100)
        red_mask = (rgb_array[:, :, 0] > 200) & (rgb_array[:, :, 1] < 100) & (rgb_array[:, :, 2] < 100)
    
    # Convert to grayscale for interpolation
    bw_array = np.array(rgb_image.convert('L')).astype(float) / 255.0
    
    # Formula: Result = (1 - intensity) * color1 + intensity * color2
    res_r = (1 - bw_array) * color1[0] + bw_array * color2[0]
    res_g = (1 - bw_array) * color1[1] + bw_array * color2[1]
    res_b = (1 - bw_array) * color1[2] + bw_array * color2[2]
    
    # Stack channels and convert back to 8-bit integers
    result_array = np.stack([res_r, res_g, res_b], axis=2).astype(np.uint8)
    
    # Restore red pixels if preserving
    if preserve_red:
        result_array[red_mask] = rgb_array[red_mask].astype(np.uint8)
    
    return Image.fromarray(result_array)

def load_shape_images(shapes_folder=args.input_folder):
    """Load all shape images from the shapes folder."""
    shape_images = {}
    
    # Find all jpeg/jpg files in the shapes folder
    image_files = glob(os.path.join(shapes_folder, '*.png')) + glob(os.path.join(shapes_folder, '*.jpeg'))
    
    for img_path in image_files:
        # Extract shape name from filename (without extension)
        shape_name = os.path.splitext(os.path.basename(img_path))[0]
        # Load the image
        shape_images[shape_name] = Image.open(img_path)
    
    return shape_images

def generate_positions(n=4, min_dist=0.8, xlim=(0.25, 3.75), ylim=(0.5, 2.5)):
    """Generate n positions with minimum distance between them."""
    positions = []
    max_attempts = 1000
    
    for i in range(n):
        for attempt in range(max_attempts):
            pos = np.random.rand(2) * [xlim[1]-xlim[0], ylim[1]-ylim[0]] + [xlim[0], ylim[0]]
            if all(np.linalg.norm(pos - p) >= min_dist for p in positions):
                positions.append(pos)
                break
    
    return np.array(positions)

def save_image(shape_types, shape_images, positions, labels, filename, size=0.3, dpi=30, use_random_colors=False):
    """Save an image with shapes at given positions and labels.
    Returns pixel positions of each shape as [[[x_min, y_min], [x_max, y_max]], ...]
    
    :param use_random_colors: If True, color all shapes with one color and background with another
    """
    # dpi = 100
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Set limits for data coordinates
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Store extents in data coordinates
    extents = []
    
    for shape_type, pos, label in zip(shape_types, positions, labels):
        # Get the shape image
        img = shape_images[shape_type]
        
        # Calculate extent for placing the image centered at position
        extent = [pos[0] - size, pos[0] + size, pos[1] - size, pos[1] + size]
        extents.append(extent)
        
        # Display the image at the specified position
        ax.imshow(img, extent=extent, aspect='auto', zorder=1)
        
        if label:
            ax.text(pos[0], pos[1] - 0.40, label, fontsize=20, ha='center', weight='bold', color='red', zorder=2)
    
    # Save with tight layout
    fig.tight_layout(pad=0)
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
    
    # Apply random colors to entire image if enabled
    if use_random_colors:
        saved_img = Image.open(filename)
        # color1 for shapes (originally black), color2 for background (originally white)
        color1 = (np.random.randint(0, 128), np.random.randint(0, 128), np.random.randint(0, 128))
        color2 = (np.random.randint(180, 255), np.random.randint(180, 255), np.random.randint(180, 255))
        colored_img = change_colors(saved_img, color1, color2, preserve_red=True)
        colored_img.save(filename)
        saved_img.close()
    
    # Reload the saved image to get its actual dimensions
    saved_img = Image.open(filename)
    actual_width, actual_height = saved_img.size
    saved_img.close()
    
    # Force a draw to ensure transforms are updated
    fig.canvas.draw()
    
    # Get the bbox that will be used for tight cropping (in inches)
    bbox_tight = fig.get_tightbbox(fig.canvas.get_renderer())
    
    # Get axis bbox in figure coordinates (0-1 scale)
    ax_bbox = ax.get_position()
    
    # Get axis bbox in inches
    fig_width_in, fig_height_in = fig.get_size_inches()
    ax_left_in = ax_bbox.x0 * fig_width_in
    ax_bottom_in = ax_bbox.y0 * fig_height_in
    ax_width_in = ax_bbox.width * fig_width_in
    ax_height_in = ax_bbox.height * fig_height_in
    
    # After tight crop, calculate where the axis lands in the final image
    ax_left_in_cropped = ax_left_in - bbox_tight.x0
    ax_bottom_in_cropped = ax_bottom_in - bbox_tight.y0
    
    # Convert to pixels in the final image
    ax_left_px = ax_left_in_cropped * dpi
    ax_bottom_px = ax_bottom_in_cropped * dpi
    ax_width_px = ax_width_in * dpi
    ax_height_px = ax_height_in * dpi
    
    # Data coordinate ranges
    x_data_range = 4.0  # 0 to 4
    y_data_range = 3.0  # 0 to 3
    
    # Pixels per data unit
    px_per_data_x = ax_width_px / x_data_range
    px_per_data_y = ax_height_px / y_data_range
    
    # Convert extents to pixel coordinates
    pixel_positions = []
    for extent in extents:
        x_min_data, x_max_data, y_min_data, y_max_data = extent
        
        # Convert data coords to pixels within the axis
        x_min_px_in_ax = x_min_data * px_per_data_x
        x_max_px_in_ax = x_max_data * px_per_data_x
        
        # Add axis offset for x
        x_min_px = int(round(ax_left_px + x_min_px_in_ax))
        x_max_px = int(round(ax_left_px + x_max_px_in_ax))
        
        # For y: matplotlib y=0 is at bottom, image y=0 is at top
        # Axis starts at ax_bottom_px from the bottom in matplotlib coords
        # In image coords, the axis top is at: actual_height - (ax_bottom_px + ax_height_px)
        ax_top_px_from_img_top = actual_height - (ax_bottom_px + ax_height_px)
        
        # Convert data y to pixels from image top
        y_min_px = int(round(ax_top_px_from_img_top + (y_data_range - y_max_data) * px_per_data_y))
        y_max_px = int(round(ax_top_px_from_img_top + (y_data_range - y_min_data) * px_per_data_y))
        
        pixel_positions.append([
            [x_min_px, y_min_px],
            [x_max_px, y_max_px]
        ])
    
    plt.close(fig)
    return pixel_positions



def create_tasks(output_folder, input_folder, num_tasks, use_random_colors=False, dpi=30):
    """
    Create visual matching tasks.
    
    :param use_random_colors: If True, randomly color each shape (default: False)
    """
    # Load shape images
    shape_images = load_shape_images(input_folder)
    print(f"Loaded {len(shape_images)} shapes: {list(shape_images.keys())}")

    os.makedirs(output_folder, exist_ok=True)

    # Use the loaded shape names as available shapes
    all_shape_types = list(shape_images.keys())

    answers = []

    for i in range(num_tasks):
        # Select 4 random shapes
        shape_types = np.random.choice(all_shape_types, min(4, len(all_shape_types)), replace=False).tolist()
        ref_idx = np.random.randint(0, len(shape_types))
        
        positions1 = generate_positions(n=len(shape_types))
        positions2 = generate_positions(n=len(shape_types))
        
        # Create ref image
        labels1 = [None] * len(shape_types)
        labels1[ref_idx] = 'REF'
        ref_pixel_positions = save_image(shape_types, shape_images, positions1, labels1, f'{output_folder}/{i:06d}_ref.jpg', use_random_colors=use_random_colors, dpi=dpi)
        
        # Create target image
        labels2 = [chr(65 + j) for j in range(len(shape_types))]
        tgt_pixel_positions = save_image(shape_types, shape_images, positions2, labels2, f'{output_folder}/{i:06d}_tgt.jpg', use_random_colors=use_random_colors, dpi=dpi)
        
        answers.append({
            'id': i,
            'answer': chr(65 + ref_idx),
            "ref_image_path": f'{output_folder}/{i:06d}_ref.jpg',
            "tgt_image_path": f'{output_folder}/{i:06d}_tgt.jpg',
            "ref_pixel_positions": ref_pixel_positions,
            "tgt_pixel_positions": tgt_pixel_positions,
            "shape_types": shape_types,
        })
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_tasks} pairs")

    with open(f'{output_folder}/answers.json', 'w') as f:
        json.dump(answers, f, indent=2)

    return answers


# Without random colors (default):
# answers = create_tasks('symbol_dataset_test', 'polygons/', 1000)

# With random colors:
answers = create_tasks(args.output_folder, args.input_folder, args.num_tasks, use_random_colors=args.use_random_colors, dpi=args.dpi)

# Other examples:
# create_tasks('symbol_dataset_train', 'symbols/', 5000, use_random_colors=False)
# create_tasks('symbol_dataset_test', 'basic_shapes/', 1000, use_random_colors=True)

