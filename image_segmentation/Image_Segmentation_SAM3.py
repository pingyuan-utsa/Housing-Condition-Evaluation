"""
SAM3 Building Component Recognition - Paper Ready Version
Output Structure:
1. Visualization with colorful mask + annotations (for paper)
2. Combined cropped components (for next step processing)
3. Pure colorful masks (for post-processing)
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Configuration
INPUT_FOLDER = r"C:\Users\dht233\sam3\testSAM"
OUTPUT_FOLDER = r"C:\Users\dht233\sam3\testSAM\results"

# Subfolders for different output types
VIS_FOLDER = os.path.join(OUTPUT_FOLDER, "01_visualizations")      # For paper
COMBINED_FOLDER = os.path.join(OUTPUT_FOLDER, "02_combined_crops") # For next step
MASK_FOLDER = os.path.join(OUTPUT_FOLDER, "03_pure_masks")         # For post-processing
INDIVIDUAL_FOLDER = os.path.join(OUTPUT_FOLDER, "04_individual_crops") # Individual crops backup

# Building components to identify
TARGETS = ["roof", "wall", "door", "window"]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.3

# BPE vocabulary path
BPE_PATH = r"C:\Users\dht233\sam3\assets\bpe_simple_vocab_16e6.txt.gz"

# Define fixed color for each category
COLOR_MAP = {
    "roof": (1.0, 0.0, 0.0),      # Red
    "wall": (0.0, 1.0, 0.0),      # Green
    "door": (0.0, 0.0, 1.0),      # Blue
    "window": (1.0, 1.0, 0.0),    # Yellow
}

def setup_torch():
    """Setup PyTorch optimization"""
    print("Setting up PyTorch...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: GPU not detected, performance may be slow")

def create_folders():
    """Create output folders with clear structure"""
    folders = [VIS_FOLDER, COMBINED_FOLDER, MASK_FOLDER, INDIVIDUAL_FOLDER]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created: {folder}")

def get_image_files():
    """Get all image files"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(INPUT_FOLDER):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    return image_files

def save_visualization_for_paper(image, inference_state, image_name, target_name):
    """
    OUTPUT 1: Visualization for paper
    - Original image with colorful mask overlay
    - Annotations: object numbers, confidence scores, bounding boxes
    - Summary statistics
    """
    masks = inference_state.get("masks", [])
    scores = inference_state.get("scores", [])
    boxes = inference_state.get("boxes", [])
    
    if len(masks) == 0:
        return
    
    base_name = os.path.splitext(image_name)[0]
    vis_filename = f"{base_name}_{target_name}_paper.png"
    vis_path = os.path.join(VIS_FOLDER, vis_filename)
    
    # Get unified color for this category
    color = COLOR_MAP.get(target_name, (1.0, 0.5, 0.0))
    color_name = {
        (1.0, 0.0, 0.0): "Red",
        (0.0, 1.0, 0.0): "Green",
        (0.0, 0.0, 1.0): "Blue",
        (1.0, 1.0, 0.0): "Yellow",
    }.get(color, "Orange")
    
    # Create figure with good resolution for paper
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(image)
    
    # === FIXED: Merge all masks first, then overlay once to prevent darkening ===
    image_array = np.array(image)
    combined_mask = np.zeros(image_array.shape[:2], dtype=float)
    
    # Merge all masks using maximum operation
    for mask in masks:
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        
        combined_mask = np.maximum(combined_mask, mask_np)
    
    # Create unified colored mask
    colored_mask = np.zeros((*combined_mask.shape, 3))
    for i in range(3):
        colored_mask[:, :, i] = combined_mask * color[i]
    
    # Overlay the combined mask ONCE (prevents multiple overlays from darkening the image)
    ax.imshow(colored_mask, alpha=0.45)
    # === END FIX ===
    
    # Draw bounding boxes and labels for each detected object
    for idx, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        
        # Add bounding box
        if torch.is_tensor(box):
            box_np = box.cpu().numpy()
        else:
            box_np = np.array(box)
        
        if len(box_np) >= 4:
            x1, y1, x2, y2 = box_np[:4]
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, 
                                fill=False, edgecolor=color, linewidth=2.5)
            ax.add_patch(rect)
        
        # Add comprehensive label
        y, x = np.where(mask_np > 0.5)
        if len(y) > 0 and len(x) > 0:
            cy, cx = int(y.mean()), int(x.mean())
            
            # Label with ID and confidence
            label_text = f"#{idx+1}\nConf: {score:.3f}"
            ax.text(cx, cy, label_text,
                   color='white', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8),
                   ha='center', va='center')
    
    # Add title with statistics
    avg_score = np.mean([s.item() if torch.is_tensor(s) else s for s in scores])
    title_text = f"{target_name.upper()} Detection Results\n"
    title_text += f"Image: {image_name} | Objects Found: {len(masks)} | "
    title_text += f"Avg Confidence: {avg_score:.3f} | Color: {color_name}"
    
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, alpha=0.45, label=f'{target_name} (n={len(masks)})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(vis_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"      [Paper] Saved: {vis_filename}")

def save_pure_colorful_mask(image_name, inference_state, target_name, image_size):
    """
    OUTPUT 3: Pure colorful mask (for post-processing)
    - Only mask, no original image
    - Same category uses same color
    - RGB format, ready for processing
    """
    masks = inference_state.get("masks", [])
    
    if len(masks) == 0:
        return
    
    # Get original image dimensions
    width, height = image_size
    
    # Create unified color mask
    color = COLOR_MAP.get(target_name, (1.0, 0.5, 0.0))
    colorful_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Merge all masks using the same color
    combined_mask = np.zeros((height, width), dtype=float)
    for mask in masks:
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        
        combined_mask = np.maximum(combined_mask, mask_np)
    
    # Apply unified color
    for i in range(3):
        colorful_mask[:, :, i] = (combined_mask * color[i] * 255).astype(np.uint8)
    
    # Save pure mask
    base_name = os.path.splitext(image_name)[0]
    mask_filename = f"{base_name}_{target_name}_mask.png"
    mask_path = os.path.join(MASK_FOLDER, mask_filename)
    
    Image.fromarray(colorful_mask).save(mask_path)
    print(f"      [Mask] Saved: {mask_filename}")
    
    return combined_mask

def crop_and_save_objects(image, image_name, inference_state, target_name):
    """
    Crop individual objects and save to individual folder
    Returns list of cropped images for combining
    """
    masks = inference_state.get("masks", [])
    scores = inference_state.get("scores", [])
    boxes = inference_state.get("boxes", [])
    
    if len(masks) == 0:
        return []
    
    image_np = np.array(image)
    base_name = os.path.splitext(image_name)[0]
    
    cropped_images = []
    
    for idx, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        # Convert mask
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        
        # Convert box
        if torch.is_tensor(box):
            box_np = box.cpu().numpy()
        else:
            box_np = np.array(box)
        
        # Get bounding box coordinates
        if len(box_np) >= 4:
            x1, y1, x2, y2 = map(int, box_np[:4])
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_np.shape[1], x2)
            y2 = min(image_np.shape[0], y2)
            
            # Crop image
            cropped = image_np[y1:y2, x1:x2]
            
            # Crop corresponding mask
            mask_cropped = mask_np[y1:y2, x1:x2]
            
            # Apply mask: background becomes white
            result = np.ones_like(cropped) * 255
            result = (result * (1 - mask_cropped[:, :, None]) + 
                     cropped * mask_cropped[:, :, None]).astype(np.uint8)
            
            # Save individual crop
            crop_filename = f"{base_name}_{target_name}_{idx+1}_individual.png"
            crop_path = os.path.join(INDIVIDUAL_FOLDER, crop_filename)
            Image.fromarray(result).save(crop_path)
            
            # Store for combining
            cropped_images.append({
                'image': result,
                'score': score.item() if torch.is_tensor(score) else score,
                'idx': idx + 1
            })
    
    print(f"      [Individual] Saved {len(cropped_images)} crops")
    return cropped_images

def combine_crops_for_next_step(cropped_list, image_name, target_name):
    """
    OUTPUT 2: Combined crops (for next step processing)
    - All objects of same category in one image
    - Grid layout with labels
    - Clean white background
    """
    if len(cropped_list) == 0:
        return
    
    base_name = os.path.splitext(image_name)[0]
    num_objects = len(cropped_list)
    
    # Calculate grid size
    cols = int(np.ceil(np.sqrt(num_objects)))
    rows = int(np.ceil(num_objects / cols))
    
    # Find max dimensions
    max_height = max([item['image'].shape[0] for item in cropped_list])
    max_width = max([item['image'].shape[1] for item in cropped_list])
    
    # Padding
    padding = 30
    label_height = 40
    
    # Create canvas
    canvas_height = rows * (max_height + label_height) + (rows + 1) * padding
    canvas_width = cols * max_width + (cols + 1) * padding
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    
    # Place images on canvas
    canvas_pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(canvas_pil)
    
    try:
        font_large = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    for i, item in enumerate(cropped_list):
        row = i // cols
        col = i % cols
        
        img = item['image']
        score = item['score']
        idx = item['idx']
        
        # Calculate position
        y_offset = row * (max_height + label_height) + (row + 1) * padding
        x_offset = col * max_width + (col + 1) * padding
        
        # Center the image in its grid cell
        y_center = y_offset + (max_height - img.shape[0]) // 2 + label_height
        x_center = x_offset + (max_width - img.shape[1]) // 2
        
        # Paste image
        img_pil = Image.fromarray(img)
        canvas_pil.paste(img_pil, (x_center, y_center))
        
        # Add label above image
        label_text = f"#{idx} (Conf: {score:.3f})"
        text_bbox = draw.textbbox((0, 0), label_text, font=font_small)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x_center + img.shape[1] // 2 - text_width // 2
        text_y = y_offset + 5
        
        draw.text((text_x, text_y), label_text, fill=(0, 0, 0), font=font_small)
    
    # Add title at the top
    title = f"{target_name.upper()} - {num_objects} objects detected"
    title_bbox = draw.textbbox((0, 0), title, font=font_large)
    title_width = title_bbox[2] - title_bbox[0]
    
    # Expand canvas for title
    final_canvas = Image.new('RGB', (canvas_width, canvas_height + 60), 'white')
    draw_final = ImageDraw.Draw(final_canvas)
    draw_final.text((canvas_width // 2 - title_width // 2, 15), 
                   title, fill=(0, 0, 0), font=font_large)
    final_canvas.paste(canvas_pil, (0, 60))
    
    # Save combined image
    combined_filename = f"{base_name}_{target_name}_combined.png"
    combined_path = os.path.join(COMBINED_FOLDER, combined_filename)
    final_canvas.save(combined_path)
    
    print(f"      [Combined] Saved: {combined_filename}")

def analyze_image(image_path, image_name, processor, model):
    """Analyze single image and generate all three outputs"""
    print(f"\nProcessing image: {image_name}")
    
    image = Image.open(image_path)
    print(f"   Size: {image.size}")
    
    for target in TARGETS:
        print(f"   Target: {target}")
        
        # Set image and prompt
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(
            state=inference_state, 
            prompt=target
        )
        
        # Check if objects found
        masks = inference_state.get("masks", [])
        if len(masks) == 0:
            print(f"      No {target} detected")
            continue
        
        # Generate all three outputs
        # Output 1: Visualization for paper
        save_visualization_for_paper(image, inference_state, image_name, target)
        
        # Output 3: Pure colorful mask
        save_pure_colorful_mask(image_name, inference_state, target, image.size)
        
        # Output 2: Combined crops (first crop individual, then combine)
        cropped_list = crop_and_save_objects(image, image_name, inference_state, target)
        combine_crops_for_next_step(cropped_list, image_name, target)

def main():
    """Main function"""
    print("=" * 70)
    print("SAM3 Building Component Recognition - Paper Ready Version")
    print("=" * 70)
    print("\nOutput Structure:")
    print("  1. Visualizations (for paper) -> 01_visualizations/")
    print("  2. Combined crops (for next step) -> 02_combined_crops/")
    print("  3. Pure masks (for post-processing) -> 03_pure_masks/")
    print("  4. Individual crops (backup) -> 04_individual_crops/")
    print("=" * 70)
    
    # Setup
    setup_torch()
    create_folders()
    
    # Get images
    image_files = get_image_files()
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {INPUT_FOLDER}")
        return
    
    print(f"\nFound {len(image_files)} images")
    print(f"\nTargets and colors:")
    for target in TARGETS:
        color = COLOR_MAP.get(target)
        color_name = {
            (1.0, 0.0, 0.0): "Red",
            (0.0, 1.0, 0.0): "Green",
            (0.0, 0.0, 1.0): "Blue",
            (1.0, 1.0, 0.0): "Yellow",
        }.get(color)
        print(f"   - {target}: {color_name}")
    
    # Load model
    print("\nLoading SAM3 model...")
    try:
        model = build_sam3_image_model(bpe_path=BPE_PATH)
        processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    # Process images
    print("\n" + "=" * 70)
    print("Processing images...")
    print("=" * 70)
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}]")
        image_path = os.path.join(INPUT_FOLDER, image_file)
        
        try:
            analyze_image(image_path, image_file, processor, model)
        except Exception as e:
            print(f"   Processing failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print("\nCheck your outputs:")
    print(f"   1. Paper figures: {VIS_FOLDER}")
    print(f"   2. Combined crops: {COMBINED_FOLDER}")
    print(f"   3. Pure masks: {MASK_FOLDER}")
    print(f"   4. Individual crops: {INDIVIDUAL_FOLDER}")
    print("=" * 70)

if __name__ == "__main__":
    main()
