# Quick Start Guide

## Which Version Should I Use?

### `app.py` - Basic Version
- Simple, lightweight
- Single dithering algorithm (Floyd-Steinberg)
- Basic controls
- Good for: Quick prototyping, learning

### `app_mvp.py` - MVP Professional Version ⭐ **RECOMMENDED**
- Full feature set
- 3 dithering algorithms (Floyd-Steinberg, Atkinson, Ordered)
- Advanced preprocessing
- Professional SVG export with DPI/unit control
- Enhanced multi-layer system
- Comprehensive tooltips and help
- Good for: Production work, professional laser cutting

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the App

### Basic Version
```bash
streamlit run app.py
```

### MVP Version (Recommended)
```bash
streamlit run app_mvp.py
```

The app will open in your browser at `http://localhost:8501`

## Feature Comparison

| Feature | app.py | app_mvp.py |
|---------|--------|------------|
| Floyd-Steinberg Dithering | ✅ | ✅ |
| Atkinson Dithering | ❌ | ✅ |
| Ordered (Bayer) Dithering | ❌ | ✅ |
| Image Preprocessing | ❌ | ✅ |
| Image Resizing | ❌ | ✅ |
| DPI Control | ❌ | ✅ |
| Unit Selection (px/mm/in) | ❌ | ✅ |
| Physical Dimension Display | ❌ | ✅ |
| Tooltips & Help | ❌ | ✅ |
| Session State & Caching | ❌ | ✅ |
| Multi-Layer Methods | 1 | 2 |
| Layer Statistics | ❌ | ✅ |
| Assembly Instructions | ❌ | ✅ |
| Material Recommendations | ❌ | ✅ |

## Recommended Workflows

### Photo Engraving on Wood
1. Upload your image
2. Select **Floyd-Steinberg** dithering
3. Enable **Auto Contrast** if image is low contrast
4. Set DPI to **300**
5. Download PNG and import into LightBurn

### Vector Cutting/Scoring
1. Upload your image
2. Adjust Canny thresholds (start with 50/150)
3. Increase **Contour Simplification** to 0.005 for smooth cuts
4. Set DPI to **96** and units to **mm**
5. Download SVG and import into LightBurn

### Multi-Layer 3D Mandala
1. Upload your image (circular/mandala designs work best)
2. Set **4-6 layers**
3. Increase **Blur** to 51-101 for organic shapes
4. Choose **Brightness Bands** method
5. Download ZIP file
6. Import layer_01.svg, layer_02.svg, etc. into LightBurn
7. Cut and stack in order

## Troubleshooting

### SVG files are the wrong size in LightBurn
- Check that DPI is set to **96** (LightBurn default)
- Verify units are set to **mm** or **in**

### Laser is jittering on edges
- Increase **Contour Simplification** slider to 0.005 or higher
- This smooths the paths using Douglas-Peucker algorithm

### Too many small contours/noise
- Increase **Minimum Contour Area** to filter small details
- For multi-layer, increase blur amount

### Image quality is poor
- Ensure source image is high resolution
- Use **Auto Contrast** for low-contrast images
- Try different dithering algorithms
- Set processing DPI to **300** or higher

## Tips for Best Results

### Dithering Choice:
- **Floyd-Steinberg**: Portraits, photos with fine detail
- **Atkinson**: High-contrast images, retro Mac aesthetic
- **Ordered 8x8**: Backgrounds, textures, repeated patterns

### Multi-Layer Depth:
- 2-3 layers: Subtle depth
- 4-6 layers: Balanced 3D effect ⭐ **RECOMMENDED**
- 7-10 layers: Maximum depth (requires precision cutting)

### Material Selection:
- **Wood**: 3mm plywood (versatile)
- **Acrylic**: 2-3mm clear or colored (modern look)
- **Cardboard**: Heavy chipboard (cheap prototypes)

## Getting Help

- **Application errors**: Check that all dependencies are installed
- **LightBurn import issues**: See [MVP_IMPROVEMENTS.md](MVP_IMPROVEMENTS.md) for DPI details
- **Feature requests**: Review future enhancements in MVP_IMPROVEMENTS.md

## Next Steps

1. Start with `app_mvp.py` (recommended)
2. Upload a test image
3. Try each of the three tabs
4. Experiment with different algorithms
5. Import results into LightBurn or Inkscape
6. Cut your first piece!

---

**Happy laser engraving!** 🔷✨
