"""
Mandala Laser Engraving Web Application - MVP Version
A comprehensive Streamlit app for converting images into laser-engraving optimized formats
with advanced features based on industry best practices
"""

import io
import zipfile
from typing import List, Tuple, Optional
import cv2
import numpy as np
import streamlit as st
import svgwrite
from PIL import Image


# ============================================================================
# DITHERING ALGORITHMS
# ============================================================================

def floyd_steinberg_dither(gray: np.ndarray) -> np.ndarray:
    """
    Floyd-Steinberg error diffusion dithering.
    Best for: General purpose, good detail preservation
    """
    img = gray.astype(np.float32)
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y, x]
            new = 255 if old > 127 else 0
            img[y, x] = new
            err = old - new
            if x + 1 < w:
                img[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += err * 3 / 16
                img[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    img[y + 1, x + 1] += err * 1 / 16
    return np.clip(img, 0, 255).astype(np.uint8)


def atkinson_dither(gray: np.ndarray) -> np.ndarray:
    """
    Atkinson dithering algorithm developed by Bill Atkinson for MacPaint.
    Best for: Better contrast in midtones, more "artistic" look
    Only propagates 75% of error, preventing wash-out in highlights
    """
    img = gray.astype(np.float32)
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y, x]
            new = 255 if old > 127 else 0
            img[y, x] = new
            err = (old - new) / 8  # Only propagate 1/8 per pixel (6/8 total)

            # Distribute to 6 neighboring pixels
            if x + 1 < w:
                img[y, x + 1] += err
            if x + 2 < w:
                img[y, x + 2] += err
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += err
                img[y + 1, x] += err
                if x + 1 < w:
                    img[y + 1, x + 1] += err
            if y + 2 < h:
                img[y + 2, x] += err

    return np.clip(img, 0, 255).astype(np.uint8)


def ordered_dither(gray: np.ndarray, matrix_size: int = 8) -> np.ndarray:
    """
    Ordered (Bayer) dithering using threshold matrix.
    Best for: Consistent patterns, GPU-friendly, no artifacts in gradients
    """
    # Bayer matrix patterns
    bayer_2 = np.array([[0, 2], [3, 1]]) / 4

    bayer_4 = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) / 16

    bayer_8 = np.array([
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ]) / 64

    matrix = {2: bayer_2, 4: bayer_4, 8: bayer_8}.get(matrix_size, bayer_4)

    h, w = gray.shape
    threshold = np.tile(matrix, (h // matrix_size + 1, w // matrix_size + 1))[:h, :w]
    normalized = gray.astype(np.float32) / 255.0
    result = (normalized > threshold).astype(np.uint8) * 255

    return result


# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

@st.cache_data
def load_and_preprocess_image(uploaded_file, target_width: Optional[int] = None,
                              target_dpi: int = 300) -> Tuple[Image.Image, dict]:
    """
    Load and optionally resize image with DPI consideration.
    Returns image and metadata dictionary.
    """
    image = Image.open(uploaded_file).convert("RGB")

    # Get original dimensions
    orig_w, orig_h = image.size

    metadata = {
        'original_width': orig_w,
        'original_height': orig_h,
        'dpi': target_dpi,
        'aspect_ratio': orig_w / orig_h
    }

    # Resize if requested
    if target_width and target_width != orig_w:
        target_height = int(target_width / metadata['aspect_ratio'])
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        metadata['resized'] = True
        metadata['new_width'] = target_width
        metadata['new_height'] = target_height
    else:
        metadata['resized'] = False

    return image, metadata


def convert_px_to_mm(pixels: int, dpi: int) -> float:
    """Convert pixels to millimeters based on DPI."""
    inches = pixels / dpi
    return inches * 25.4


def convert_px_to_inches(pixels: int, dpi: int) -> float:
    """Convert pixels to inches based on DPI."""
    return pixels / dpi


# ============================================================================
# SVG GENERATION WITH DPI SUPPORT
# ============================================================================

def contours_to_svg(contours: List[np.ndarray], width: int, height: int,
                   dpi: int = 96, units: str = 'px', stroke_width: float = 0.1) -> str:
    """
    Create SVG with proper DPI and unit specifications for LightBurn compatibility.

    Args:
        contours: List of contour arrays
        width: Width in pixels
        height: Height in pixels
        dpi: Target DPI (96 for LightBurn default, 72 for Inkscape)
        units: 'px', 'mm', or 'in'
        stroke_width: Line width for cutting paths
    """
    # Calculate dimensions in target units
    if units == 'mm':
        w_units = convert_px_to_mm(width, dpi)
        h_units = convert_px_to_mm(height, dpi)
        scale = w_units / width
    elif units == 'in':
        w_units = convert_px_to_inches(width, dpi)
        h_units = convert_px_to_inches(height, dpi)
        scale = w_units / width
    else:  # px
        w_units = width
        h_units = height
        scale = 1.0

    # Create SVG with proper dimensions
    dwg = svgwrite.Drawing(
        size=(f"{w_units}{units}", f"{h_units}{units}"),
        viewBox=f"0 0 {width} {height}",
        profile='full'
    )

    # Add proper namespaces for compatibility
    dwg.attribs['xmlns'] = 'http://www.w3.org/2000/svg'
    dwg.attribs['xmlns:xlink'] = 'http://www.w3.org/1999/xlink'

    # Add metadata for better software compatibility
    dwg.attribs['version'] = '1.1'

    # Add paths
    for cnt in contours:
        if len(cnt) < 2:
            continue
        pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]

        # Create path data
        path_data = f"M {pts[0][0]},{pts[0][1]}"
        for x, y in pts[1:]:
            path_data += f" L {x},{y}"
        path_data += " Z"

        dwg.add(dwg.path(
            d=path_data,
            fill="none",
            stroke="black",
            stroke_width=stroke_width
        ))

    return dwg.tostring()


def find_and_simplify_contours(binary: np.ndarray, epsilon_factor: float = 0.002,
                               min_area: float = 10.0) -> List[np.ndarray]:
    """
    Find and simplify contours with area filtering.

    Args:
        binary: Binary image
        epsilon_factor: Simplification factor (higher = more simplified)
        min_area: Minimum contour area to keep
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    simplified = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        eps = max(1.0, peri * epsilon_factor)
        approx = cv2.approxPolyDP(c, eps, True)
        simplified.append(approx)

    return simplified


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def svg_bytes_from_contours(contours: List[np.ndarray], w: int, h: int,
                            dpi: int = 96, units: str = 'px') -> bytes:
    """Convert contours to SVG bytes with DPI/unit support."""
    svg_text = contours_to_svg(contours, w, h, dpi=dpi, units=units)
    return svg_text.encode("utf-8")


def make_layer_preview(contours: List[np.ndarray], w: int, h: int) -> Image.Image:
    """Create preview image from contours."""
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    if len(contours) > 0:
        cv2.drawContours(canvas, contours, -1, (0, 0, 0), 1)
    return Image.fromarray(canvas)


def create_zip_from_svg_list(svgs: List[Tuple[str, bytes]]) -> bytes:
    """Create ZIP file from list of SVG files."""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in svgs:
            zf.writestr(name, data)
    mem.seek(0)
    return mem.read()


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Mandala Laser Engraving App - MVP",
        page_icon="🔷",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None

    # Header
    st.title("🔷 Mandala Laser Engraving App - MVP")
    st.markdown("**Professional-grade image processing for laser engraving with multi-layer support**")

    # Sidebar for global settings
    with st.sidebar:
        st.header("⚙️ Global Settings")

        st.subheader("Output Configuration")
        output_dpi = st.selectbox(
            "SVG DPI Standard",
            options=[72, 90, 96, 300],
            index=2,
            help="96 DPI = LightBurn default | 72 DPI = Inkscape/Adobe | 300 DPI = High resolution"
        )

        output_units = st.selectbox(
            "Output Units",
            options=['px', 'mm', 'in'],
            index=1,
            help="Measurement units for SVG files. 'mm' recommended for laser cutters."
        )

        st.subheader("Image Preprocessing")
        enable_resize = st.checkbox("Enable Image Resizing", value=False)

        if enable_resize:
            resize_width = st.number_input(
                "Target Width (pixels)",
                min_value=100,
                max_value=5000,
                value=800,
                step=50,
                help="Resize image to this width. Height will maintain aspect ratio."
            )
        else:
            resize_width = None

        target_dpi = st.number_input(
            "Processing DPI",
            min_value=72,
            max_value=600,
            value=300,
            step=10,
            help="DPI for calculations. Higher = larger physical size."
        )

        st.markdown("---")
        st.markdown("### 📊 Current Session")
        if st.session_state.processed_image:
            st.success("✓ Image loaded")
        else:
            st.info("Upload an image to begin")

    # File upload
    uploaded = st.file_uploader(
        "📁 Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )

    if uploaded is None:
        st.info("👆 Upload an image to begin processing. Use the tabs below to choose your processing mode.")

        # Show quick guide
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 📸 Photo Engraving")
            st.markdown("Convert photos to 1-bit black & white using advanced dithering algorithms")
        with col2:
            st.markdown("#### ✏️ Vector Scoring")
            st.markdown("Extract edges and convert to SVG paths for precise laser cutting")
        with col3:
            st.markdown("#### 🏔️ Multi-Layer 3D")
            st.markdown("Create stacked layers for stunning 3D mandala effects")

        return

    # Load and preprocess image
    image, metadata = load_and_preprocess_image(uploaded, resize_width, target_dpi)
    st.session_state.processed_image = image

    w, h = image.size

    # Show image info
    with st.expander("📐 Image Information", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Width", f"{w} px")
        with col2:
            st.metric("Height", f"{h} px")
        with col3:
            physical_w = convert_px_to_mm(w, target_dpi)
            st.metric("Physical Width", f"{physical_w:.1f} mm")
        with col4:
            physical_h = convert_px_to_mm(h, target_dpi)
            st.metric("Physical Height", f"{physical_h:.1f} mm")

    # Convert to grayscale for processing
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "📸 Photo Engraving (Dithering)",
        "✏️ Vector Scoring (Edge Detection)",
        "🏔️ Multi-Layer Topographic"
    ])

    # ========================================================================
    # TAB 1: PHOTO ENGRAVING
    # ========================================================================
    with tab1:
        st.header("Photo Engraving — Advanced Dithering")
        st.markdown("Convert your image to pure 1-bit black & white for photo engraving on wood, acrylic, leather, etc.")

        # Dithering algorithm selection
        dither_method = st.selectbox(
            "Dithering Algorithm",
            options=["Floyd-Steinberg", "Atkinson", "Ordered (Bayer)"],
            help="**Floyd-Steinberg**: Best overall quality, good detail\n\n"
                 "**Atkinson**: Better contrast, more artistic, developed for MacPaint\n\n"
                 "**Ordered**: Consistent patterns, good for textures"
        )

        if dither_method == "Ordered (Bayer)":
            matrix_size = st.select_slider(
                "Bayer Matrix Size",
                options=[2, 4, 8],
                value=4,
                help="Larger matrix = finer grain. 8x8 recommended for most images."
            )

        # Optional preprocessing
        with st.expander("🔧 Advanced Preprocessing", expanded=False):
            apply_contrast = st.checkbox("Auto Contrast Enhancement", value=True)
            if apply_contrast:
                contrast_strength = st.slider("Contrast Strength", 1.0, 3.0, 1.5, 0.1)

            apply_sharpen = st.checkbox("Sharpen Image", value=False)
            if apply_sharpen:
                sharpen_amount = st.slider("Sharpen Amount", 0.0, 2.0, 1.0, 0.1)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Dithered Result")

            # Apply preprocessing
            processed_gray = gray.copy()

            if apply_contrast:
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=contrast_strength, tileGridSize=(8,8))
                processed_gray = clahe.apply(processed_gray)

            if apply_sharpen:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * sharpen_amount / 9
                processed_gray = cv2.filter2D(processed_gray, -1, kernel)
                processed_gray = np.clip(processed_gray, 0, 255).astype(np.uint8)

            # Apply selected dithering
            with st.spinner(f"Applying {dither_method} dithering..."):
                if dither_method == "Floyd-Steinberg":
                    dithered = floyd_steinberg_dither(processed_gray)
                elif dither_method == "Atkinson":
                    dithered = atkinson_dither(processed_gray)
                else:  # Ordered
                    dithered = ordered_dither(processed_gray, matrix_size)

            pil_dither = Image.fromarray(dithered).convert("L")
            st.image(pil_dither, use_container_width=True, clamp=True)

            # Download button
            buf = io.BytesIO()
            pil_dither.save(buf, format="PNG", dpi=(target_dpi, target_dpi))
            buf.seek(0)

            st.download_button(
                label="⬇️ Download Dithered PNG",
                data=buf,
                file_name=f"engraving_{dither_method.lower()}.png",
                mime="image/png",
                help=f"Resolution: {w}x{h} pixels @ {target_dpi} DPI"
            )

    # ========================================================================
    # TAB 2: VECTOR SCORING
    # ========================================================================
    with tab2:
        st.header("Vector Scoring — Edge Detection → SVG")
        st.markdown("Extract edges and convert to simplified vector paths for laser cutting/scoring")

        # Edge detection parameters
        col_param1, col_param2 = st.columns(2)

        with col_param1:
            min_threshold = st.slider(
                "Canny Min Threshold",
                0, 255, 50, 5,
                help="Lower threshold for edge detection. Lower = more edges."
            )
            blur_kernel = st.slider(
                "Gaussian Blur Kernel",
                1, 51, 5, 2,
                help="Noise reduction. Higher = smoother but less detail."
            )

        with col_param2:
            max_threshold = st.slider(
                "Canny Max Threshold",
                1, 500, 150, 5,
                help="Upper threshold for edge detection. Ratio 1:2 or 1:3 recommended."
            )
            epsilon_factor = st.slider(
                "Contour Simplification",
                0.0, 0.02, 0.002, 0.0005,
                help="Higher = smoother paths (CRITICAL for preventing laser jitter)"
            )

        min_area = st.slider(
            "Minimum Contour Area (filter noise)",
            0.0, 100.0, 10.0, 5.0,
            help="Remove tiny contours smaller than this area"
        )

        # Process
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        edges = cv2.Canny(blurred, min_threshold, max_threshold)

        # Find and simplify contours
        contours = find_and_simplify_contours(
            edges,
            epsilon_factor=epsilon_factor,
            min_area=min_area
        )

        # Display
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Edge Detection Preview")
            st.image(edges, use_container_width=True, caption="Canny edges (raster)")

        with col2:
            st.subheader("Vectorized Contours")
            preview_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
            if len(contours) > 0:
                cv2.drawContours(preview_canvas, contours, -1, (0, 0, 0), 1)
            st.image(preview_canvas, use_container_width=True, caption=f"{len(contours)} simplified contours")

        # Generate SVG
        svg_data = svg_bytes_from_contours(contours, w, h, dpi=output_dpi, units=output_units)

        # Download
        st.download_button(
            label=f"⬇️ Download SVG ({output_dpi} DPI, {output_units})",
            data=svg_data,
            file_name="vector_scoring.svg",
            mime="image/svg+xml",
            help=f"Optimized for LightBurn/Inkscape | {len(contours)} contours"
        )

        # Stats
        st.info(f"📊 **Contour Statistics**: {len(contours)} paths | DPI: {output_dpi} | Units: {output_units}")

    # ========================================================================
    # TAB 3: MULTI-LAYER TOPOGRAPHIC
    # ========================================================================
    with tab3:
        st.header("Multi-Layer Topographic — Stacked 3D Mandala Design")
        st.markdown("Create depth-layered designs where each layer represents a brightness range, perfect for stacked 3D effects")

        # Layer parameters
        col_layer1, col_layer2 = st.columns(2)

        with col_layer1:
            num_layers = st.slider(
                "Number of Layers",
                2, 10, 4,
                help="More layers = finer depth graduation. 4-6 recommended."
            )
            heavy_blur = st.slider(
                "Blur Amount (organic shapes)",
                3, 201, 51, 2,
                help="Higher = more organic flowing shapes. Lower = sharper edges."
            )

        with col_layer2:
            layer_method = st.selectbox(
                "Layer Generation Method",
                ["Brightness Bands", "Cumulative Threshold"],
                help="**Brightness Bands**: Each layer is a separate brightness range\n\n"
                     "**Cumulative Threshold**: Each layer includes all darker areas (traditional topo)"
            )
            invert_layers = st.checkbox(
                "Invert (light → dark stacking)",
                value=False,
                help="Check to stack lightest layer first instead of darkest"
            )

        # Process
        with st.spinner("Generating layers..."):
            blurred = cv2.GaussianBlur(gray, (heavy_blur, heavy_blur), 0)

            if invert_layers:
                blurred = 255 - blurred

            # Show blurred preview
            with st.expander("🔍 View Blurred Base Image", expanded=False):
                st.image(blurred, use_container_width=True, clamp=True)

            # Generate layers
            svgs = []
            previews = []
            layer_info = []

            if layer_method == "Brightness Bands":
                # Each layer is a separate brightness band
                thresholds = np.linspace(0, 255, num_layers + 1)

                for i in range(num_layers):
                    lo = int(thresholds[i])
                    hi = int(thresholds[i + 1])

                    # Create mask for this brightness range
                    mask = cv2.inRange(blurred, lo, hi)

                    # Morphological operations to clean up
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    # Find contours
                    contour_list = find_and_simplify_contours(
                        mask,
                        epsilon_factor=0.003,
                        min_area=20.0
                    )

                    # Generate SVG
                    svg_bytes = svg_bytes_from_contours(
                        contour_list, w, h,
                        dpi=output_dpi,
                        units=output_units
                    )

                    layer_name = f"layer_{i+1:02d}_band_{lo}-{hi}.svg"
                    svgs.append((layer_name, svg_bytes))
                    previews.append(make_layer_preview(contour_list, w, h))
                    layer_info.append({
                        'layer': i + 1,
                        'range': f"{lo}-{hi}",
                        'contours': len(contour_list)
                    })

            else:  # Cumulative Threshold
                # Each layer includes all areas darker than threshold
                thresholds = np.linspace(255, 0, num_layers + 1)

                for i in range(num_layers):
                    threshold_val = int(thresholds[i + 1])

                    # Create mask for all areas darker than threshold
                    _, mask = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

                    # Clean up
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                    # Find contours (use RETR_EXTERNAL for outer contours only)
                    contours_raw, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                    )

                    # Simplify
                    contour_list = []
                    for c in contours_raw:
                        if cv2.contourArea(c) < 20.0:
                            continue
                        peri = cv2.arcLength(c, True)
                        eps = max(1.0, peri * 0.003)
                        approx = cv2.approxPolyDP(c, eps, True)
                        contour_list.append(approx)

                    # Generate SVG
                    svg_bytes = svg_bytes_from_contours(
                        contour_list, w, h,
                        dpi=output_dpi,
                        units=output_units
                    )

                    layer_name = f"layer_{i+1:02d}_threshold_{threshold_val}.svg"
                    svgs.append((layer_name, svg_bytes))
                    previews.append(make_layer_preview(contour_list, w, h))
                    layer_info.append({
                        'layer': i + 1,
                        'threshold': threshold_val,
                        'contours': len(contour_list)
                    })

        # Display layer previews
        st.subheader(f"📐 Layer Previews ({num_layers} layers)")

        # Show layers in a grid
        cols_per_row = min(5, num_layers)
        rows_needed = (num_layers + cols_per_row - 1) // cols_per_row

        for row in range(rows_needed):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                layer_idx = row * cols_per_row + col_idx
                if layer_idx < num_layers:
                    with cols[col_idx]:
                        st.image(
                            previews[layer_idx],
                            caption=f"Layer {layer_idx + 1}",
                            use_container_width=True
                        )

        # Layer information table
        with st.expander("📊 Layer Details", expanded=False):
            import pandas as pd
            df = pd.DataFrame(layer_info)
            st.dataframe(df, use_container_width=True)

        # Create ZIP
        zip_bytes = create_zip_from_svg_list(svgs)

        # Download
        st.download_button(
            label=f"⬇️ Download All {num_layers} Layers (ZIP)",
            data=zip_bytes,
            file_name=f"mandala_layers_{num_layers}x.zip",
            mime="application/zip",
            help=f"Contains {num_layers} SVG files | {output_dpi} DPI | {output_units} units"
        )

        st.success(f"✅ Successfully generated {num_layers} layers! Download the ZIP file and import into LightBurn.")

        # Assembly tips
        with st.expander("🔨 Assembly Tips", expanded=False):
            st.markdown("""
            ### Stacking Order
            - **Layer 1** (darkest/bottom) should be cut first
            - Stack layers in numerical order (1, 2, 3...)
            - Use spacers or adhesive between layers for depth

            ### Material Recommendations
            - Wood: 3mm plywood or MDF
            - Acrylic: 2-3mm clear or colored
            - Cardboard: Heavy chipboard for prototypes

            ### Cutting Settings
            - **Cut** the outer contour of each layer
            - Consider **engraving** the next layer's outline as an alignment guide
            - Test cut Layer 1 first to verify material thickness
            """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p><strong>🔷 Mandala Laser Engraving App - MVP Edition</strong></p>
            <p>Optimized for LightBurn, Inkscape, and professional laser cutters</p>
            <p style='font-size: 0.9em;'>
                Features: Floyd-Steinberg/Atkinson/Ordered Dithering •
                Canny Edge Detection with Anti-Jitter •
                Multi-Layer 3D Topographic Stacking •
                DPI/Unit Control
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
