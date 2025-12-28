# MVP Improvements - Research & Implementation Summary

## Overview
This document details the research-driven improvements made to transform the basic Mandala app into an MVP-grade professional tool.

## Research Sources & Key Findings

### 1. Laser Engraving Best Practices

**Sources:**
- [3 steps to the perfect photo laser engraving](https://www.troteclaser.com/en-us/helpcenter/software/jobcontrol/photo-laser-engraving)
- [Laser Engraving: Prepare Images in 6 Steps](https://www.imag-r.com/)
- [rastercarve - PyPI](https://pypi.org/project/rastercarve/)

**Key Findings:**
- **Minimum 300 DPI** required for quality laser engraving
- Lower resolutions (333-500 DPI) create more sculptural effects
- Different dithering algorithms produce different engraving qualities:
  - **Error Diffusion** (Floyd-Steinberg): Chaotic raster with dense dots in dark areas
  - **Ordered Dithering**: Organized raster with larger dots in dark areas
- Images need good contrast and exposure for best results

### 2. Multi-Layer Topographic Design

**Sources:**
- [Laser-Cut Topography: Vertical - Harvard GSD](https://wiki.harvard.edu/confluence/display/fabricationlab/Laser-Cut+Topography:+Vertical)
- [Easy 3D Topographical Maps With Slicer](https://www.instructables.com/Easy-3D-Topographical-Maps/)
- [Making a Laser-Cut Topo Map](https://theshamblog.com/making-a-laser-cut-topo-map-the-design-phase/)

**Key Findings:**
- Two main approaches:
  1. **Brightness Bands**: Each layer represents a separate brightness range
  2. **Cumulative Threshold**: Each layer includes all darker areas (traditional topographic)
- Material optimization: smaller layers can nest inside larger ones
- Assembly features: engrave next layer outline on current layer for alignment
- Vertical contour lines should match material thickness spacing

### 3. Streamlit UX Best Practices

**Sources:**
- [5 steps to build a Data Web App MVP](https://medium.com/the-streamlit-teacher/5-steps-to-build-a-data-web-app-mvp-with-python-and-streamlit-5f6abe8332d)
- [Streamlit Hands-On: Features and Tips](https://towardsdatascience.com/streamlit-hands-on-features-and-tips-for-enhanced-app-user-experience-aef7be8035fa/)
- [Crafting Interactive Image Processing Apps](https://medium.com/@daniela.vorkel/crafting-interactive-image-processing-apps-for-jpg-jpeg-enhancement-with-streamlit-and-python-c36b907b3f52)

**Key Findings:**
- **Sidebar organization**: Keep controls in expandable sidebar
- **Tooltips are critical**: Use `help=` parameter on all widgets
- **Performance**: Use `@st.cache_data` for expensive operations
- **Session state**: Track processing state across reruns
- **Real-time preview**: Essential for image processing apps
- **Dark mode consideration**: Better for viewing image variations

### 4. LightBurn SVG Compatibility

**Sources:**
- [SVG Import scaling issue - LightBurn Forum](https://forum.lightburnsoftware.com/t/svg-import-scaling-issue/46787)
- [Correct Import Settings - LightBurn Forum](https://forum.lightburnsoftware.com/t/correct-import-settings/136779)
- [Explicitly set user units on SVG import](https://lightburn.fider.io/posts/138/explictly-set-user-units-on-svg-import)

**Key Findings:**
- **DPI Standards**:
  - LightBurn default: **96 DPI**
  - Inkscape/Adobe: **72 DPI**
  - 96 vs 72 DPI causes 1.333x scaling factor
- **Units matter**: Explicitly specify 'mm', 'in', or 'px' in SVG
- **Namespace requirements**: Full SVG 1.1 spec with proper xmlns
- Common issue: Scaling mismatch between export and import software

### 5. Dithering Algorithm Comparison

**Sources:**
- [Ditherpunk - Monochrome Image Dithering](https://surma.dev/things/ditherpunk/)
- [Image Dithering: Eleven Algorithms](https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html)
- [Atkinson Dithering](https://beyondloom.com/blog/dither.html)

**Algorithm Comparison:**

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Floyd-Steinberg** | Good quality, fast, standard | Can create artifacts in large flat areas | General purpose, portraits |
| **Atkinson** | Better contrast in midtones, "artistic" look | Loses detail in highlights/shadows (only propagates 75% error) | High-contrast images, retro Mac aesthetic |
| **Ordered (Bayer)** | GPU-friendly, consistent patterns, no artifacts | Less organic than error diffusion | Textures, backgrounds, parallel processing |

## MVP Improvements Implemented

### 1. Multiple Dithering Algorithms ✅
- **Floyd-Steinberg**: Original, best general-purpose
- **Atkinson**: Better midtone contrast, artistic look
- **Ordered (Bayer)**: 2x2, 4x4, 8x8 matrices for different grain sizes

### 2. Advanced Image Preprocessing ✅
- **Auto-contrast**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Sharpening**: Configurable kernel-based sharpening
- **Resizing**: Optional image resizing with aspect ratio preservation
- **Caching**: `@st.cache_data` for image loading

### 3. Professional SVG Export ✅
- **DPI Control**: 72, 90, 96, 300 DPI options
- **Unit Specification**: px, mm, in support
- **Proper Namespaces**: Full SVG 1.1 spec
- **Physical Dimensions**: Automatic calculation from DPI

### 4. Enhanced Multi-Layer System ✅
- **Two Generation Methods**:
  - Brightness Bands (original)
  - Cumulative Threshold (traditional topographic)
- **Inversion Option**: Light-to-dark stacking
- **Better Morphology**: Improved noise reduction
- **Area Filtering**: Remove tiny contours
- **Assembly Tips**: Built-in documentation

### 5. Superior User Experience ✅
- **Sidebar Configuration**: Global settings organized
- **Tooltips Everywhere**: Every control has help text
- **Session State**: Tracks processing state
- **Image Metrics**: Show dimensions and physical size
- **Layer Statistics**: Detailed contour information
- **Expander Sections**: Collapsible advanced options
- **Progress Indicators**: Spinners for long operations

### 6. Quality of Life Features ✅
- **Image Information Panel**: Shows px and mm dimensions
- **Contour Count Display**: Real-time feedback
- **Layer Preview Grid**: 5 columns max for clarity
- **Layer Details Table**: Pandas DataFrame with statistics
- **Assembly Instructions**: Built-in tutorial
- **Material Recommendations**: Guidance for materials

## Files Structure

```
Mandala/
├── app.py                  # Original basic version
├── app_mvp.py             # New MVP version (USE THIS)
├── requirements.txt        # Updated dependencies
├── README.md              # User documentation
└── MVP_IMPROVEMENTS.md    # This file (developer docs)
```

## Performance Optimizations

1. **Caching**: Image loading cached with `@st.cache_data`
2. **Session State**: Prevents redundant processing
3. **Efficient Algorithms**: Optimized contour simplification
4. **Smart Defaults**: Research-based default values

## Usage Recommendations

### For Photo Engraving:
- Use **Floyd-Steinberg** for portraits and detailed photos
- Use **Atkinson** for high-contrast artistic images
- Use **Ordered 8x8** for textured backgrounds
- Enable **Auto Contrast** for low-contrast photos
- Set DPI to **300** for best quality

### For Vector Scoring:
- Start with default thresholds (50/150)
- Increase **Contour Simplification** to 0.005-0.01 for smoother paths
- Use **Minimum Area** filter to remove noise
- Export at **96 DPI** for LightBurn, **72 DPI** for Inkscape
- Use **mm** units for laser cutters

### For Multi-Layer:
- Use **4-6 layers** for balanced depth
- **Blur 51-101** for organic mandala shapes
- **Blur 11-31** for sharper architectural designs
- Use **Cumulative Threshold** for traditional topo maps
- Use **Brightness Bands** for abstract art
- Export at **96 DPI, mm units** for best compatibility

## Testing Checklist

- [x] Floyd-Steinberg dithering produces correct output
- [x] Atkinson dithering works with proper error distribution
- [x] Ordered dithering with all matrix sizes (2, 4, 8)
- [x] Image resizing maintains aspect ratio
- [x] DPI settings affect physical dimensions correctly
- [x] SVG files import correctly into LightBurn at 96 DPI
- [x] SVG files import correctly into Inkscape at 72 DPI
- [x] Multi-layer brightness bands generate distinct layers
- [x] Multi-layer cumulative threshold creates nested contours
- [x] ZIP download contains all layers
- [x] Tooltips display on all controls
- [x] Session state persists across interactions
- [x] Image caching improves performance

## Known Limitations

1. **Processing Speed**: Dithering is CPU-intensive for large images (>2000px)
2. **Memory Usage**: Multi-layer processing stores all previews in memory
3. **Browser Limits**: Very large images may cause browser slowdown
4. **SVG Complexity**: Thousands of contours may slow down laser software

## Future Enhancements (Post-MVP)

### Potential Additions:
- [ ] Jarvis-Judice-Ninke dithering
- [ ] Stucki dithering
- [ ] Burkes dithering
- [ ] GPU acceleration for dithering
- [ ] Batch processing multiple images
- [ ] Custom color palette support
- [ ] G-code generation
- [ ] Material database with settings
- [ ] Preview 3D stacking visualization
- [ ] Export directly to printer/cutter
- [ ] Halftone patterns for color engraving
- [ ] Vector trace (potrace algorithm)

## Changelog

### v2.0 (MVP) - 2025-12-28
- Added Atkinson and Ordered dithering
- Implemented image preprocessing (contrast, sharpen)
- Added DPI and unit controls
- Improved SVG compatibility
- Enhanced multi-layer algorithm
- Added comprehensive tooltips
- Implemented session state and caching
- Added image metrics display
- Created assembly documentation
- Optimized contour simplification

### v1.0 (Basic) - 2025-12-27
- Floyd-Steinberg dithering only
- Basic Canny edge detection
- Simple multi-layer threshold
- Minimal UI controls

## References

All research sources are cited throughout this document with hyperlinks to original materials.

---

**Note**: Use `app_mvp.py` for the production-ready MVP version with all improvements.
