# 🔷 Mandala Laser Engraving App

A powerful Python-based web application that converts images into laser-engraving optimized formats. Perfect for creating stunning mandala designs with up to 10 stacked layers for 3D depth effects.

## Features

### 1. 📸 Photo Engraving Mode (Dithering)
- Converts images to pure 1-bit black and white
- Uses Floyd-Steinberg error-diffusion algorithm
- Perfect for photo engraving on wood, acrylic, or leather
- Outputs high-quality PNG files

### 2. ✏️ Vector Scoring Mode (Edge Detection)
- Extracts edges using Canny edge detection
- Converts to vector SVG format
- **Critical Feature:** Applies `cv2.approxPolyDP` contour simplification to prevent laser jitter
- Adjustable threshold controls for precise edge detection
- Compatible with LightBurn and Inkscape

### 3. 🏔️ Multi-Layer Topographic Mode
- Creates stacked 3D designs with 2-10 layers
- Darker layers are cut separately for depth effect
- Organic shape generation through heavy Gaussian blur
- Outputs ZIP file with individual SVG layers
- Perfect for creating stunning mandala art pieces

## Tech Stack

- **Language:** Python 3.9+
- **UI Framework:** Streamlit
- **Image Processing:** OpenCV (cv2), NumPy, Pillow (PIL)
- **Vector Generation:** svgwrite

## 🚀 Quick Start

We provide **three versions** of the application:

1. **`app.py`** - Basic version with core features
2. **`app_mvp.py`** - MVP version with professional features
3. **`app_enhanced.py`** - ⭐ **ENHANCED version with 20 layers + beginner guides** (RECOMMENDED)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run ENHANCED version (recommended for beginners and advanced users)
streamlit run app_enhanced.py

# OR run MVP version
streamlit run app_mvp.py

# OR run basic version
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

📖 **See [QUICK_START.md](QUICK_START.md) for detailed comparison and workflow guides**

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Using the App

1. **Upload an Image**
   - Click "Browse files" or drag and drop your image
   - Supported formats: PNG, JPG, JPEG, BMP, TIFF

2. **Choose a Processing Mode** (via tabs)

   **Photo Engraving:**
   - View the dithered result
   - Download as PNG for raster engraving

   **Vector Scoring:**
   - Adjust Canny edge detection thresholds using sliders
   - Adjust Gaussian blur for noise reduction
   - Increase contour simplification to prevent laser jitter
   - Download as SVG for vector cutting/scoring

   **Multi-Layer Topographic:**
   - Set number of layers (2-10)
   - Adjust blur amount for organic vs. sharp shapes
   - Preview all layers side-by-side
   - Download ZIP file containing all layer SVG files

3. **Import to Your Laser Software**
   - Open downloaded SVG files in LightBurn, Inkscape, or similar
   - All SVG files have proper namespaces and headers
   - Ready for immediate use with laser cutters

## How It Works

### Floyd-Steinberg Dithering
The algorithm converts grayscale images to pure black and white by distributing quantization error to neighboring pixels, creating the illusion of gray tones through dot patterns.

### Edge Detection with Anti-Jitter
1. Gaussian blur reduces noise
2. Canny edge detection finds edges
3. **Crucial:** `cv2.approxPolyDP` simplifies contours using the Douglas-Peucker algorithm
4. This prevents the laser from "jittering" on jagged pixel edges
5. Results in smooth, efficient cutting paths

### Multi-Layer Stacking
1. Heavy blur creates organic shapes
2. Image brightness is divided into threshold ranges
3. Each layer represents a brightness band
4. Darker layers are cut first and stacked
5. Creates stunning 3D depth effect

## File Outputs

- **Photo Engraving:** `dithered.png` - 1-bit black and white raster image
- **Vector Scoring:** `edges.svg` - Vector paths for laser cutting
- **Multi-Layer:** `layers.zip` - Contains `layer_1.svg`, `layer_2.svg`, etc.

## Tips for Best Results

### Photo Engraving
- Use high-contrast images for best dithering results
- Works great for portraits and detailed photos

### Vector Scoring
- Start with min threshold: 50, max threshold: 150
- Increase contour simplification (epsilon) to 0.005-0.01 for smoother paths
- Higher blur values work better for noisy images

### Multi-Layer Topographic
- Use 4-6 layers for balanced 3D effect
- Increase blur (51-101) for organic, flowing shapes
- Decrease blur (11-31) for sharper, more defined edges
- Test cut layer 1 first to ensure proper material thickness

## Laser Cutter Compatibility

The SVG files generated include:
- Proper XML namespace declarations
- Standard SVG 1.1 specification
- Compatible with:
  - LightBurn
  - Inkscape
  - Adobe Illustrator
  - Most laser cutter software

## Troubleshooting

**Issue:** SVG files won't open in LightBurn
- **Solution:** Files are already optimized with proper namespaces. Ensure LightBurn is updated to latest version.

**Issue:** Laser is jittering on edges
- **Solution:** Increase the "Contour simplification" slider in Vector Scoring mode.

**Issue:** Too many small details in multi-layer mode
- **Solution:** Increase the blur kernel size to create larger, more organic shapes.

**Issue:** App won't start
- **Solution:** Ensure all dependencies are installed: `pip install -r requirements.txt`

## Project Structure

```
Mandala/
│
├── app.py                  # Basic version (original)
├── app_mvp.py             # MVP version (professional)
├── app_enhanced.py        # ⭐ ENHANCED version with 20 layers (RECOMMENDED)
├── requirements.txt        # Python dependencies
├── README.md              # This file (overview)
├── QUICK_START.md         # Quick start guide
├── MVP_IMPROVEMENTS.md    # MVP research documentation
└── ENHANCED_FEATURES.md   # Enhanced version features & research
```

## What's New in Enhanced Version 🎓

The **ENHANCED version** (`app_enhanced.py`) includes everything from MVP plus:

### 🆕 Enhanced-Only Features:
- **20-Layer Support**: Increased from 10 to 20 layers for complex projects
- **Material Calculator**: Pre-loaded materials + custom thickness calculator
- **Measurement Overlays**: Toggle-able rulers on previews
- **Alignment Marks**: SVG crosshairs for precision multi-layer cutting
- **Beginner Mode**: Toggle extra help and warnings
- **Tutorial Guide**: Comprehensive expandable beginner's guide
- **Stack Height Calc**: Automatic calculation of total project height
- **Safety Warnings**: PVC alerts, ventilation reminders, fire safety
- **Material Database**: 8 pre-loaded materials with thickness values
- **Welcome Screen**: Visual mode previews with descriptions

### 🎨 All MVP Features Included:
- **3 Dithering Algorithms**: Floyd-Steinberg, Atkinson, Ordered (Bayer)
- **Professional Export**: DPI control (72/90/96/300), unit selection (mm/in/px)
- **Image Preprocessing**: Auto-contrast (CLAHE), sharpening, resizing
- **Two Layer Methods**: Brightness Bands + Cumulative Threshold
- **Performance**: Session state and caching
- **Enhanced UX**: Tooltips, help text, organized sidebar

---

### 📖 Documentation:
- **[ENHANCED_FEATURES.md](ENHANCED_FEATURES.md)** - Complete feature list with research sources
- **[MVP_IMPROVEMENTS.md](MVP_IMPROVEMENTS.md)** - MVP research and technical details
- **[QUICK_START.md](QUICK_START.md)** - Version comparison and workflows

## Contributing

This is a single-file application designed for simplicity and ease of use. Feel free to modify `app.py` to suit your specific needs.

## License

This project is open source and available for personal and commercial use.

## Support

For issues or questions, please refer to the Streamlit documentation at https://docs.streamlit.io

---

**Happy Laser Engraving!** 🔷✨
