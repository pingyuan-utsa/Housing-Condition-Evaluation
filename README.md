# SAM3 Building Component Recognition

A Python tool for automated detection and segmentation of building components (roof, wall, door, window) using the SAM3 (Segment Anything Model 3) framework. This tool generates publication-ready visualizations with annotations, combined crops for further processing, and pure masks for post-processing.

## Features

- **Automated Detection**: Detects and segments four building components: roof, wall, door, and window
- **Multiple Output Formats**:
  - High-resolution visualizations with annotations (for papers/presentations)
  - Combined cropped components in grid layout (for next-step processing)
  - Pure color-coded masks (for post-processing)
  - Individual component crops (for backup)
- **Color-Coded Categories**: 
  - Roof (Red)
  - Wall (Green)
  - Door (Blue)
  - Window (Yellow)
- **Comprehensive Annotations**: Bounding boxes, confidence scores, object IDs, and statistics
- **Batch Processing**: Process multiple images automatically
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Anaconda or Miniconda
- Windows/Linux/macOS

## Installation

### 1. Clone SAM3 Repository

Follow the official [SAM3 repository](https://github.com/facebookresearch/sam3) to set up the base environment:

```bash
# Clone the SAM3 repository
git clone https://github.com/facebookresearch/sam3.git
cd sam3

# Create conda environment
conda create -n sam3 python=3.10 -y
conda activate sam3

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install SAM3
pip install -e .
```

### 2. Windows-Specific Fix

**Important**: For Windows users, install the Windows-compatible Triton package:

```bash
pip install triton-windows
```

Without this, you will encounter the error: `ModuleNotFoundError: No module named 'triton'`

### 3. Install Additional Dependencies

```bash
pip install matplotlib pillow numpy
```

### 4. Download BPE Vocabulary File

Download the required BPE vocabulary file from the SAM3 repository:
- File: `bpe_simple_vocab_16e6.txt.gz`
- Location: Place it in your working directory or update the `BPE_PATH` in the script


## Configuration

Before running the script, update the following paths in `Image_Segmentation_SAM3.py`:

```python
# Line 18-19: Set your input and output folders
INPUT_FOLDER = r"C:\path\to\your\images"
OUTPUT_FOLDER = r"C:\path\to\your\results"

# Line 34: Set the BPE vocabulary path
BPE_PATH = r"C:\path\to\bpe_simple_vocab_16e6.txt.gz"

# Optional configurations:
# Line 28: Detection targets
TARGETS = ["roof", "wall", "door", "window"]

# Line 31: Confidence threshold (0.0-1.0)
CONFIDENCE_THRESHOLD = 0.3

# Line 37-42: Color mapping for each category
COLOR_MAP = {
    "roof": (1.0, 0.0, 0.0),      # Red
    "wall": (0.0, 1.0, 0.0),      # Green
    "door": (0.0, 0.0, 1.0),      # Blue
    "window": (1.0, 1.0, 0.0),    # Yellow
}
```

## Usage

### Running the Script

1. **Open Anaconda Prompt** (or terminal)

2. **Activate the environment**:
   ```bash
   conda activate sam3
   ```

3. **Navigate to your working directory**:
   ```bash
   cd path/to/sam3-building-recognition
   ```

4. **Run the script**:
   ```bash
   python Image_Segmentation_SAM3.py
   ```

### Input Requirements

- Place your building images in the `INPUT_FOLDER`
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Images should clearly show the building components you want to detect

## Output Structure

The script generates four types of outputs in organized subfolders:

```
results/
├── 01_visualizations/          # Publication-ready figures
│   ├── image1_roof_paper.png
│   ├── image1_door_paper.png
│   └── ...
├── 02_combined_crops/          # Grid layouts of detected components
│   ├── image1_roof_combined.png
│   ├── image1_door_combined.png
│   └── ...
├── 03_pure_masks/              # Color-coded segmentation masks
│   ├── image1_roof_mask.png
│   ├── image1_door_mask.png
│   └── ...
└── 04_individual_crops/        # Individual component crops
    ├── image1_roof_1_individual.png
    ├── image1_roof_2_individual.png
    └── ...
```

### Output Details

#### 1. Visualizations (01_visualizations/)
- **Purpose**: Publication-ready figures for papers and presentations
- **Features**:
  - Original image with semi-transparent colored mask overlay
  - Bounding boxes around each detected component
  - Object IDs and confidence scores
  - Title with detection statistics
  - Legend showing category and count
- **Format**: High-resolution PNG (300 DPI)
- **Naming**: `{image_name}_{component}_paper.png`

#### 2. Combined Crops (02_combined_crops/)
- **Purpose**: Grid layout of all detected components for next-step processing
- **Features**:
  - All components of the same category in one image
  - White background
  - Grid layout with labels showing object ID and confidence
  - Title with total count
- **Format**: PNG
- **Naming**: `{image_name}_{component}_combined.png`

#### 3. Pure Masks (03_pure_masks/)
- **Purpose**: Segmentation masks for post-processing and analysis
- **Features**:
  - Black background
  - Color-coded regions (one color per category)
  - No original image or annotations
  - Same dimensions as input image
- **Format**: RGB PNG
- **Naming**: `{image_name}_{component}_mask.png`

#### 4. Individual Crops (04_individual_crops/)
- **Purpose**: Backup of individual component images
- **Features**:
  - Each detected object as a separate image
  - White background
  - Cropped to bounding box with mask applied
- **Format**: PNG
- **Naming**: `{image_name}_{component}_{ID}_individual.png`

## Code Logic

### Main Processing Flow

```
1. Setup
   - Initialize PyTorch with GPU optimization
   - Create output folder structure
   - Load SAM3 model with BPE vocabulary

2. Image Loading
   - Scan INPUT_FOLDER for image files
   - Validate file formats

3. For Each Image:
   - Load image with PIL
   - For Each Target Component (roof, wall, door, window):
       - Set image in SAM3 processor
       - Set text prompt (component name)
       - Run inference to get masks, scores, and boxes
       - If components detected:
           - Generate visualization with annotations
           - Generate pure color mask
           - Crop individual components
           - Combine crops in grid layout
       - Save all outputs to respective folders

4. Summary
   - Print statistics and output locations
```

### Key Functions

#### `save_visualization_for_paper()`
Generates publication-ready visualizations:
- Displays original image as background
- Merges all masks for same category into one combined mask
- Overlays combined mask with transparency (alpha=0.45)
- **Fix Applied**: Prevents image darkening by combining masks before overlay instead of overlaying each mask separately
- Draws bounding boxes and labels for each detected object
- Adds title, legend, and statistics

#### `save_pure_colorful_mask()`
Creates pure segmentation masks:
- Generates black background image
- Merges all masks for same category
- Colors the masked regions with category-specific color
- Saves as RGB image for further processing

#### `crop_and_save_objects()`
Extracts individual components:
- Uses bounding boxes to crop each detected object
- Applies mask to set background to white
- Saves each component as separate image
- Returns list of cropped images for combining

#### `combine_crops_for_next_step()`
Creates grid layout:
- Calculates optimal grid dimensions
- Places all cropped components on white canvas
- Adds labels with object ID and confidence score
- Includes title with total count

### Detection Pipeline

1. **Image Input**: PIL Image loaded from file
2. **SAM3 Processing**: 
   - Image encoder extracts features
   - Text prompt ("roof", "wall", etc.) guides segmentation
   - Model generates masks, confidence scores, and bounding boxes
3. **Post-Processing**:
   - Filter by confidence threshold (default: 0.3)
   - Convert tensors to numpy arrays
   - Generate multiple output formats
4. **Visualization**: Matplotlib for annotations and rendering

## Troubleshooting

### Common Issues

**1. "No module named 'triton'" Error**
- **Solution**: Install `triton-windows` for Windows systems:
  ```bash
  pip install triton-windows
  ```

**2. "CUDA out of memory" Error**
- **Solution**: Reduce batch size or use smaller images
- Try processing images one at a time
- Close other GPU-intensive applications

**3. "No images found" Error**
- **Solution**: Check that `INPUT_FOLDER` path is correct
- Verify image files have supported extensions
- Ensure images are not in subdirectories

**4. Model Loading Failed**
- **Solution**: Verify `BPE_PATH` points to correct vocabulary file
- Check that SAM3 is properly installed: `pip show sam3`
- Ensure you have sufficient disk space for model weights

**5. Low Detection Quality**
- **Solution**: Adjust `CONFIDENCE_THRESHOLD` (lower = more detections, higher = fewer but more confident)
- Try different image resolutions
- Ensure images have good lighting and clear building features

**6. Font Display Issues on Linux**
- **Solution**: Install system fonts or let script use default font
  ```bash
  sudo apt-get install fonts-dejavu
  ```

## Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed for significant speed improvements
2. **Batch Processing**: Process multiple images in one run rather than individually
3. **Image Size**: Very large images (>4K) may require more memory; consider resizing if needed
4. **Confidence Threshold**: Higher threshold (e.g., 0.5) = faster processing, fewer detections

## Technical Specifications

- **Model**: SAM3 (Segment Anything Model 3)
- **Input**: RGB images (JPG, PNG, BMP, TIFF)
- **Output Resolution**: 300 DPI for visualizations
- **Confidence Threshold**: 0.3 (configurable)
- **Color Depth**: 24-bit RGB
- **GPU Memory**: ~8GB recommended for optimal performance

## Citation

If you use this code in your research, please cite the original SAM3 paper:

```bibtex
@article{sam3_2024,
  title={SAM3: Segment Anything Model 3},
  author={Meta AI Research},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI Research for the SAM3 model
- PyTorch team for the deep learning framework
- All contributors to the open-source computer vision community

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Note**: This tool is designed for research and academic purposes. Results should be validated for production use.
