"""
SAM3 Building Component Recognition - Enhanced Version
1. Same category objects use unified color (all windows are yellow)
2. Automatically crop and save each identified object
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Configuration
INPUT_FOLDER = r"C:\Users\dht233\sam3\testSAM"
OUTPUT_FOLDER = r"C:\Users\dht233\sam3\testSAM\results"
CROP_FOLDER = r"C:\Users\dht233\sam3\testSAM\cropped"

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

# Function Definitions

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
    """Create output folders"""
    for folder in [OUTPUT_FOLDER, CROP_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Using existing folder: {folder}")

def get_image_files():
    """Get all image files"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(INPUT_FOLDER):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    return image_files

def save_masks_separately(image_name, inference_state, target_name):
    """Save each object's mask separately as grayscale image"""
    masks = inference_state.get("masks", [])
    scores = inference_state.get("scores", [])
    
    if len(masks) == 0:
        print(f"   {target_name}: No objects found")
        return
    
    print(f"   {target_name}: Found {len(masks)} objects")
    
    for idx, (mask, score) in enumerate(zip(masks, scores)):
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        
        # Save grayscale mask
        mask_img = (mask_np * 255).astype(np.uint8)
        base_name = os.path.splitext(image_name)[0]
        mask_filename = f"{base_name}_{target_name}_{idx+1}_score{score:.2f}.png"
        mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)
        
        Image.fromarray(mask_img).save(mask_path)
        print(f"      Saved mask: {mask_filename}")

def save_colorful_mask(image_name, inference_state, target_name):
    """
    Save colorful mask - same category objects use unified color
    This file can be directly used for subsequent processing like cropping
    """
    masks = inference_state.get("masks", [])
    scores = inference_state.get("scores", [])
    
    if len(masks) == 0:
        return
    
    # Get original image dimensions
    first_mask = masks[0]
    if torch.is_tensor(first_mask):
        mask_np = first_mask.cpu().numpy()
    else:
        mask_np = np.array(first_mask)
    
    if len(mask_np.shape) > 2:
        mask_np = mask_np.squeeze()
    
    height, width = mask_np.shape
    
    # Create colorful mask with unified color in RGB format
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
    
    # Save colorful mask
    base_name = os.path.splitext(image_name)[0]
    colorful_filename = f"{base_name}_{target_name}_colorful_mask.png"
    colorful_path = os.path.join(OUTPUT_FOLDER, colorful_filename)
    
    Image.fromarray(colorful_mask).save(colorful_path)
    print(f"      Saved colorful mask: {colorful_filename}")
    
    return colorful_mask, combined_mask

def crop_objects(image, image_name, inference_state, target_name):
    """
    Crop each object based on mask
    Save as individual image files
    """
    masks = inference_state.get("masks", [])
    scores = inference_state.get("scores", [])
    boxes = inference_state.get("boxes", [])
    
    if len(masks) == 0:
        return
    
    image_np = np.array(image)
    base_name = os.path.splitext(image_name)[0]
    
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
        # Box format may be [x1, y1, x2, y2] or other formats
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
            
            # Apply mask: background becomes white or transparent
            # Option 1: Keep original image, background becomes white
            result = np.ones_like(cropped) * 255  # White background
            result = (result * (1 - mask_cropped[:, :, None]) + 
                     cropped * mask_cropped[:, :, None]).astype(np.uint8)
            
            # Save cropped result
            crop_filename = f"{base_name}_{target_name}_{idx+1}_cropped.png"
            crop_path = os.path.join(CROP_FOLDER, crop_filename)
            Image.fromarray(result).save(crop_path)
            print(f"      Cropped: {crop_filename}")

def save_visualization(image, inference_state, image_name, target_name):
    """Save visualization result with same category objects using unified color"""
    masks = inference_state.get("masks", [])
    scores = inference_state.get("scores", [])
    
    if len(masks) == 0:
        return
    
    base_name = os.path.splitext(image_name)[0]
    vis_filename = f"{base_name}_{target_name}_visualization.png"
    vis_path = os.path.join(OUTPUT_FOLDER, vis_filename)
    
    # Get unified color for this category
    color = COLOR_MAP.get(target_name, (1.0, 0.5, 0.0))
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # All objects of the same category use the same color
    for idx, (mask, score) in enumerate(zip(masks, scores)):
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)
        
        if len(mask_np.shape) > 2:
            mask_np = mask_np.squeeze()
        
        # Create colorful mask with unified color
        colored_mask = np.zeros((*mask_np.shape, 3))
        for i in range(3):
            colored_mask[:, :, i] = mask_np * color[i]
        
        plt.imshow(colored_mask, alpha=0.5)
        
        # Add labels
        y, x = np.where(mask_np > 0.5)
        if len(y) > 0 and len(x) > 0:
            cy, cx = int(y.mean()), int(x.mean())
            plt.text(cx, cy, f"{idx+1}\n{score:.2f}", 
                    color='white', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                    ha='center', va='center')
    
    plt.title(f"{image_name} - {target_name} ({len(masks)} found) - Color: {target_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(vis_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"      Saved visualization: {vis_filename}")

def analyze_image(image_path, image_name, processor, model):
    """Analyze single image"""
    print(f"\nProcessing image: {image_name}")
    
    image = Image.open(image_path)
    print(f"   Size: {image.size}")
    
    for target in TARGETS:
        color_name = {
            (1.0, 0.0, 0.0): "Red",
            (0.0, 1.0, 0.0): "Green",
            (0.0, 0.0, 1.0): "Blue",
            (1.0, 1.0, 0.0): "Yellow",
        }.get(COLOR_MAP.get(target), "Orange")
        print(f"   Identifying: {target} (color: {color_name})")
        
        # Set image and prompt
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(
            state=inference_state, 
            prompt=target
        )
        
        # Save various outputs
        save_masks_separately(image_name, inference_state, target)
        save_colorful_mask(image_name, inference_state, target)
        save_visualization(image, inference_state, image_name, target)
        crop_objects(image, image_name, inference_state, target)

def main():
    """Main function"""
    print("=" * 70)
    print("SAM3 Building Component Recognition - Enhanced Version")
    print("   - Same category objects use unified color")
    print("   - Automatically crop identified objects")
    print("=" * 70)
    
    # Step 1: Setup environment
    setup_torch()
    create_folders()
    
    # Step 2: Get image list
    image_files = get_image_files()
    
    if len(image_files) == 0:
        print(f"Error: No image files found in {INPUT_FOLDER}")
        return
    
    print(f"\nFound {len(image_files)} images")
    print(f"Identification targets and colors:")
    for target in TARGETS:
        color = COLOR_MAP.get(target, (1.0, 0.5, 0.0))
        color_name = {
            (1.0, 0.0, 0.0): "Red",
            (0.0, 1.0, 0.0): "Green",
            (0.0, 0.0, 1.0): "Blue",
            (1.0, 1.0, 0.0): "Yellow",
        }.get(color, "Orange")
        print(f"   - {target}: {color_name}")
    
    # Step 3: Load model
    print("\nLoading SAM3 model...")
    try:
        model = build_sam3_image_model(bpe_path=BPE_PATH)
        processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    
    # Step 4: Process each image
    print("\n" + "=" * 70)
    print("Starting image processing...")
    print("=" * 70)
    
    for idx, image_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}]")
        image_path = os.path.join(INPUT_FOLDER, image_file)
        
        try:
            analyze_image(image_path, image_file, processor, model)
        except Exception as e:
            print(f"   Processing failed: {e}")
            continue
    
    # Step 5: Complete
    print("\n" + "=" * 70)
    print("All images processed successfully")
    print(f"\nOutput locations:")
    print(f"   - Masks: {OUTPUT_FOLDER}")
    print(f"   - Cropped images: {CROP_FOLDER}")
    print("\nOutput file descriptions:")
    print("   - *_mask.png: Grayscale mask")
    print("   - *_colorful_mask.png: Colorful mask (same category, same color)")
    print("   - *_visualization.png: Visualization overlaid on original image")
    print("   - *_cropped.png: Cropped objects (white background)")
    print("=" * 70)

if __name__ == "__main__":
    main()
