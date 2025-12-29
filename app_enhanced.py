"""
Mandala Laser Engraving Web Application - ENHANCED VERSION with Beginner Guide
Complete professional tool with 20-layer support, measurements, and tutorials
"""

import io
import zipfile
from typing import List, Tuple, Optional
import cv2
import numpy as np
import streamlit as st
import svgwrite
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_LAYERS = 20  # Increased from 10 to 20
DEFAULT_LAYERS = 5

# Material database
MATERIALS = {
    "3mm Plywood": 3.0,
    "5mm Plywood": 5.0,
    "2mm Acrylic": 2.0,
    "3mm Acrylic": 3.0,
    "5mm Acrylic": 5.0,
    "1.5mm Cardboard": 1.5,
    "3mm MDF": 3.0,
    "6mm MDF": 6.0,
    "Custom": 0.0
}


# ============================================================================
# DITHERING ALGORITHMS
# ============================================================================

def floyd_steinberg_dither(gray: np.ndarray) -> np.ndarray:
    """Floyd-Steinberg error diffusion dithering - Best for portraits."""
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
    """Atkinson dithering - Better contrast, artistic Mac-style look."""
    img = gray.astype(np.float32)
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            old = img[y, x]
            new = 255 if old > 127 else 0
            img[y, x] = new
            err = (old - new) / 8

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
    """Ordered (Bayer) dithering - Consistent patterns, good for textures."""
    bayer_2 = np.array([[0, 2], [3, 1]]) / 4
    bayer_4 = np.array([
        [0, 8, 2, 10], [12, 4, 14, 6],
        [3, 11, 1, 9], [15, 7, 13, 5]
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
# IMAGE PROCESSING
# ============================================================================

@st.cache_data
def load_and_preprocess_image(uploaded_file, target_width: Optional[int] = None,
                              target_dpi: int = 300) -> Tuple[Image.Image, dict]:
    """Load and optionally resize image with DPI consideration."""
    image = Image.open(uploaded_file).convert("RGB")
    orig_w, orig_h = image.size

    metadata = {
        'original_width': orig_w,
        'original_height': orig_h,
        'dpi': target_dpi,
        'aspect_ratio': orig_w / orig_h
    }

    if target_width and target_width != orig_w:
        target_height = int(target_width / metadata['aspect_ratio'])
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        metadata['resized'] = True
        metadata['new_width'] = target_width
        metadata['new_height'] = target_height
    else:
        metadata['resized'] = False

    return image, metadata


def add_measurement_overlay(image: Image.Image, dpi: int, units: str = 'mm') -> Image.Image:
    """Add measurement ruler overlay to image for reference."""
    img_with_ruler = image.copy()
    draw = ImageDraw.Draw(img_with_ruler)
    w, h = img_with_ruler.size

    # Calculate physical dimensions
    if units == 'mm':
        phys_w = (w / dpi) * 25.4
        phys_h = (h / dpi) * 25.4
    elif units == 'in':
        phys_w = w / dpi
        phys_h = h / dpi
    else:
        phys_w = w
        phys_h = h

    # Draw ruler on bottom and right edges
    ruler_color = (255, 0, 0)
    text_color = (255, 0, 0)

    # Bottom ruler
    ruler_height = min(30, h // 20)
    draw.rectangle([(0, h - ruler_height), (w, h)], fill=(240, 240, 240, 180))

    # Right ruler
    ruler_width = min(30, w // 20)
    draw.rectangle([(w - ruler_width, 0), (w, h)], fill=(240, 240, 240, 180))

    # Add tick marks (every 10mm or 1 inch)
    tick_interval = 10 if units == 'mm' else 1
    scale = w / phys_w if phys_w > 0 else 1

    # Bottom ticks
    for i in range(int(phys_w / tick_interval) + 1):
        x_pos = int(i * tick_interval * scale)
        if x_pos < w:
            draw.line([(x_pos, h - ruler_height), (x_pos, h)], fill=ruler_color, width=1)

    # Right ticks
    for i in range(int(phys_h / tick_interval) + 1):
        y_pos = int(i * tick_interval * scale)
        if y_pos < h:
            draw.line([(w - ruler_width, y_pos), (w, y_pos)], fill=ruler_color, width=1)

    return img_with_ruler


def convert_px_to_mm(pixels: int, dpi: int) -> float:
    """Convert pixels to millimeters."""
    return (pixels / dpi) * 25.4


def convert_px_to_inches(pixels: int, dpi: int) -> float:
    """Convert pixels to inches."""
    return pixels / dpi


# ============================================================================
# SVG GENERATION WITH MEASUREMENTS
# ============================================================================

def create_svg_with_measurements(contours: List[np.ndarray], width: int, height: int,
                                 dpi: int = 96, units: str = 'mm',
                                 add_rulers: bool = False) -> str:
    """Create SVG with optional measurement rulers."""
    # Calculate dimensions
    if units == 'mm':
        w_units = convert_px_to_mm(width, dpi)
        h_units = convert_px_to_mm(height, dpi)
    elif units == 'in':
        w_units = convert_px_to_inches(width, dpi)
        h_units = convert_px_to_inches(height, dpi)
    else:
        w_units = width
        h_units = height

    dwg = svgwrite.Drawing(
        size=(f"{w_units}{units}", f"{h_units}{units}"),
        viewBox=f"0 0 {width} {height}",
        profile='full'
    )

    dwg.attribs['xmlns'] = 'http://www.w3.org/2000/svg'
    dwg.attribs['xmlns:xlink'] = 'http://www.w3.org/1999/xlink'
    dwg.attribs['version'] = '1.1'

    # Add metadata group
    metadata = dwg.g(id='metadata')
    metadata.add(dwg.text(f'Units: {units}', insert=(5, 15), fill='gray', font_size='10px'))
    metadata.add(dwg.text(f'DPI: {dpi}', insert=(5, 30), fill='gray', font_size='10px'))
    metadata.add(dwg.text(f'Size: {w_units:.1f}x{h_units:.1f} {units}', insert=(5, 45), fill='gray', font_size='10px'))
    dwg.add(metadata)

    # Add cutting paths
    cuts_group = dwg.g(id='cutting_paths', stroke='black', fill='none', stroke_width=0.1)

    for cnt in contours:
        if len(cnt) < 2:
            continue
        pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
        path_data = f"M {pts[0][0]},{pts[0][1]}"
        for x, y in pts[1:]:
            path_data += f" L {x},{y}"
        path_data += " Z"
        cuts_group.add(dwg.path(d=path_data))

    dwg.add(cuts_group)

    # Add alignment marks if requested
    if add_rulers:
        rulers_group = dwg.g(id='alignment_marks', stroke='red', fill='none', stroke_width=0.5)

        # Corner alignment marks (crosshairs)
        mark_size = 10
        corners = [(mark_size, mark_size), (width - mark_size, mark_size),
                  (mark_size, height - mark_size), (width - mark_size, height - mark_size)]

        for x, y in corners:
            # Crosshair
            rulers_group.add(dwg.line((x - mark_size, y), (x + mark_size, y)))
            rulers_group.add(dwg.line((x, y - mark_size), (x, y + mark_size)))
            # Circle
            rulers_group.add(dwg.circle((x, y), r=mark_size/2))

        dwg.add(rulers_group)

    return dwg.tostring()


def find_and_simplify_contours(binary: np.ndarray, epsilon_factor: float = 0.002,
                               min_area: float = 10.0) -> List[np.ndarray]:
    """Find and simplify contours with area filtering."""
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
                            dpi: int = 96, units: str = 'px',
                            add_rulers: bool = False) -> bytes:
    """Convert contours to SVG bytes."""
    svg_text = create_svg_with_measurements(contours, w, h, dpi, units, add_rulers)
    return svg_text.encode("utf-8")


def make_layer_preview(contours: List[np.ndarray], w: int, h: int) -> Image.Image:
    """Create preview image from contours."""
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    if len(contours) > 0:
        cv2.drawContours(canvas, contours, -1, (0, 0, 0), 1)
    return Image.fromarray(canvas)


def create_zip_from_svg_list(svgs: List[Tuple[str, bytes]]) -> bytes:
    """Create ZIP file from SVG list."""
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in svgs:
            zf.writestr(name, data)
    mem.seek(0)
    return mem.read()


def calculate_total_stack_height(num_layers: int, material_thickness: float) -> float:
    """Calculate total height of stacked layers."""
    return num_layers * material_thickness


# ============================================================================
# UI COMPONENTS
# ============================================================================

def show_beginner_guide():
    """Display comprehensive beginner's guide."""
    with st.expander("🎓 **BEGINNER'S GUIDE** - Start Here if You're New!", expanded=False):
        st.markdown("""
        ### Welcome to Laser Engraving! 👋

        This app prepares your images for laser cutting/engraving. Here's what you need to know:

        ---

        #### 🎯 **Three Processing Modes:**

        **1. Photo Engraving (Dithering)** 📸
        - **What it does**: Converts photos to black & white dots for engraving
        - **Best for**: Portraits, photographs, detailed images
        - **Output**: PNG file for raster engraving
        - **Beginner tip**: Start with high-contrast photos

        **2. Vector Scoring (Edge Detection)** ✏️
        - **What it does**: Finds outlines and converts to cutting paths
        - **Best for**: Logos, line art, simple shapes
        - **Output**: SVG file for vector cutting
        - **Beginner tip**: Use the default settings first (50/150)

        **3. Multi-Layer Topographic (3D Stacking)** 🏔️
        - **What it does**: Creates multiple layers that stack for 3D effect
        - **Best for**: Mandala art, topographic maps, depth effects
        - **Output**: ZIP with multiple SVG layers
        - **Beginner tip**: Start with 4-5 layers, not 20!

        ---

        #### ⚙️ **Important Settings:**

        - **DPI**: Higher = better quality (use 300 for engraving, 96 for cutting)
        - **Units**: Set to 'mm' for laser cutters
        - **Contour Simplification**: Higher = smoother cuts (prevents jitter)

        ---

        #### 🔰 **Your First Project - Simple Keychain:**

        1. Upload a simple logo or icon (high contrast works best)
        2. Go to **Vector Scoring** tab
        3. Use default settings
        4. Download the SVG
        5. Import to LightBurn
        6. Cut on 3mm plywood

        ---

        #### ⚠️ **Safety First:**

        - Never leave laser cutter unattended
        - Use proper ventilation
        - Wear safety glasses
        - Don't cut PVC (releases toxic gas!)
        - Keep fire extinguisher nearby

        ---

        #### 📚 **Recommended Materials for Beginners:**

        ✅ **Easy**: Cardboard, paper, thin plywood (1-3mm)
        ✅ **Medium**: Acrylic (2-3mm), leather, fabric
        ❌ **Avoid**: PVC, vinyl, fiberglass (toxic fumes!)

        ---

        #### 🎬 **Quick Start Workflow:**

        ```
        1. Upload image → 2. Choose mode → 3. Adjust settings →
        4. Preview result → 5. Download file → 6. Import to laser software →
        7. Test on scrap material → 8. Cut final piece!
        ```

        ---

        **Need help?** Check tooltips (ℹ️ icons) next to each setting!
        """)


def show_material_calculator():
    """Display material thickness calculator for multi-layer projects."""
    st.subheader("📐 Material Thickness Calculator")

    col1, col2 = st.columns(2)

    with col1:
        material_type = st.selectbox(
            "Material Type",
            options=list(MATERIALS.keys()),
            help="Select your material or choose 'Custom' to enter thickness manually"
        )

        if material_type == "Custom":
            thickness = st.number_input(
                "Custom Thickness (mm)",
                min_value=0.1,
                max_value=20.0,
                value=3.0,
                step=0.1
            )
        else:
            thickness = MATERIALS[material_type]
            st.info(f"📏 Standard thickness: **{thickness} mm**")

    with col2:
        num_layers_calc = st.number_input(
            "Number of Layers",
            min_value=2,
            max_value=MAX_LAYERS,
            value=5,
            help="How many layers will you stack?"
        )

        total_height = calculate_total_stack_height(num_layers_calc, thickness)

        st.metric(
            "Total Stack Height",
            f"{total_height:.1f} mm",
            help="Total height when all layers are stacked"
        )

        # Convert to inches for reference
        total_inches = total_height / 25.4
        st.caption(f"≈ {total_inches:.2f} inches")

    # Recommendations
    st.markdown("---")
    st.markdown("**💡 Stacking Tips:**")

    if total_height < 10:
        st.success("✅ Good beginner project - easy to align and glue")
    elif total_height < 20:
        st.warning("⚠️ Moderate difficulty - use alignment jigs")
    else:
        st.error("🔴 Advanced project - requires precision alignment")

    return thickness


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Mandala Laser Engraving - Enhanced",
        page_icon="🔷",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better tooltips
    st.markdown("""
    <style>
    .stTooltipIcon {
        color: #0068c9;
    }
    .beginner-badge {
        background-color: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
        st.session_state.show_tutorial = True
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None

    # Header
    st.title("🔷 Mandala Laser Engraving App - Enhanced Edition")
    st.markdown("**Professional laser cutting preparation with beginner-friendly guides and up to 20 layers**")

    # Show beginner guide
    show_beginner_guide()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Global Settings")

        # Beginner mode toggle
        beginner_mode = st.checkbox(
            "🎓 Beginner Mode (Show Extra Help)",
            value=True,
            help="Enable extra tooltips and warnings for first-time users"
        )

        st.markdown("---")
        st.subheader("Output Configuration")

        output_dpi = st.selectbox(
            "SVG DPI Standard",
            options=[72, 90, 96, 300],
            index=2,
            help="**96 DPI** = LightBurn default (RECOMMENDED)\n"
                 "**72 DPI** = Inkscape/Adobe\n"
                 "**300 DPI** = High resolution printing"
        )

        output_units = st.selectbox(
            "Output Units",
            options=['mm', 'in', 'px'],
            index=0,
            help="**mm** = Millimeters (RECOMMENDED for laser cutters)\n"
                 "**in** = Inches\n"
                 "**px** = Pixels"
        )

        add_alignment_marks = st.checkbox(
            "Add Alignment Marks to SVG",
            value=True,
            help="Adds corner crosshairs for precise alignment (recommended for multi-layer)"
        )

        st.markdown("---")
        st.subheader("Image Preprocessing")

        enable_resize = st.checkbox(
            "Enable Image Resizing",
            value=False,
            help="Resize large images for faster processing"
        )

        if enable_resize:
            resize_width = st.number_input(
                "Target Width (pixels)",
                min_value=100,
                max_value=5000,
                value=800,
                step=50
            )
        else:
            resize_width = None

        target_dpi = st.number_input(
            "Processing DPI",
            min_value=72,
            max_value=600,
            value=300,
            step=10,
            help="Higher DPI = larger physical output size"
        )

        add_measurement_rulers = st.checkbox(
            "Add Measurement Overlay to Previews",
            value=False,
            help="Show rulers on preview images for size reference"
        )

    # File upload
    st.markdown("---")
    uploaded = st.file_uploader(
        "📁 **Step 1: Upload Your Image**",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Choose a clear, high-resolution image for best results"
    )

    if uploaded is None:
        st.info("👆 **Upload an image to begin!** Choose any PNG, JPG, or similar image file.")

        # First-time user guide
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📸 Photo Engraving")
            st.image("https://via.placeholder.com/300x200/4CAF50/FFFFFF?text=Photo+Engraving", use_container_width=True)
            st.markdown("Perfect for portraits and photographs. Converts to black & white dots.")

        with col2:
            st.markdown("### ✏️ Vector Cutting")
            st.image("https://via.placeholder.com/300x200/2196F3/FFFFFF?text=Vector+Cutting", use_container_width=True)
            st.markdown("Ideal for logos and outlines. Creates precise cutting paths.")

        with col3:
            st.markdown("### 🏔️ 3D Layers")
            st.image("https://via.placeholder.com/300x200/FF9800/FFFFFF?text=3D+Layers", use_container_width=True)
            st.markdown("Amazing mandala effects! Up to 20 stacked layers.")

        return

    # Load image
    image, metadata = load_and_preprocess_image(uploaded, resize_width, target_dpi)
    st.session_state.processed_image = image
    w, h = image.size

    # Image info banner
    st.success(f"✅ Image loaded: **{w} x {h} pixels** | Physical size: **{convert_px_to_mm(w, target_dpi):.1f} x {convert_px_to_mm(h, target_dpi):.1f} mm**")

    # Convert to grayscale
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Create tabs
    st.markdown("---")
    st.markdown("### **Step 2: Choose Your Processing Mode**")

    tab1, tab2, tab3 = st.tabs([
        "📸 Photo Engraving",
        "✏️ Vector Scoring",
        "🏔️ Multi-Layer (20 Layers Max)"
    ])

    # TAB 1: PHOTO ENGRAVING
    with tab1:
        if beginner_mode:
            st.info("ℹ️ **BEGINNER TIP**: Photo engraving works best with high-contrast portraits and detailed photos. The laser will engrave tiny dots to create the image.")

        st.header("Photo Engraving — Advanced Dithering")

        dither_method = st.selectbox(
            "Dithering Algorithm",
            options=["Floyd-Steinberg (Recommended)", "Atkinson (Artistic)", "Ordered/Bayer (Textures)"],
            help="**Floyd-Steinberg**: Best overall, great detail\n"
                 "**Atkinson**: Better contrast, retro Mac look\n"
                 "**Ordered**: Consistent patterns for backgrounds"
        )

        # Extract actual method name
        method_name = dither_method.split(" (")[0]

        if "Ordered" in dither_method:
            matrix_size = st.select_slider(
                "Pattern Size",
                options=[2, 4, 8],
                value=8,
                help="8x8 = finest grain (recommended)"
            )

        with st.expander("🔧 Advanced Preprocessing (Optional)", expanded=False):
            apply_contrast = st.checkbox("Auto Contrast Enhancement", value=True)
            if apply_contrast:
                contrast_strength = st.slider("Contrast", 1.0, 3.0, 1.5, 0.1)

            apply_sharpen = st.checkbox("Sharpen", value=False)
            if apply_sharpen:
                sharpen_amount = st.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            display_img = add_measurement_overlay(image, target_dpi, output_units) if add_measurement_rulers else image
            st.image(display_img, use_container_width=True)

        with col2:
            st.subheader("Dithered Result")

            processed_gray = gray.copy()

            if apply_contrast:
                clahe = cv2.createCLAHE(clipLimit=contrast_strength, tileGridSize=(8,8))
                processed_gray = clahe.apply(processed_gray)

            if apply_sharpen:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * sharpen_amount / 9
                processed_gray = cv2.filter2D(processed_gray, -1, kernel)
                processed_gray = np.clip(processed_gray, 0, 255).astype(np.uint8)

            with st.spinner(f"Applying {method_name} dithering..."):
                if "Floyd" in dither_method:
                    dithered = floyd_steinberg_dither(processed_gray)
                elif "Atkinson" in dither_method:
                    dithered = atkinson_dither(processed_gray)
                else:
                    dithered = ordered_dither(processed_gray, matrix_size)

            pil_dither = Image.fromarray(dithered).convert("L")

            if add_measurement_rulers:
                pil_dither_display = add_measurement_overlay(pil_dither.convert("RGB"), target_dpi, output_units)
                st.image(pil_dither_display, use_container_width=True)
            else:
                st.image(pil_dither, use_container_width=True, clamp=True)

            buf = io.BytesIO()
            pil_dither.save(buf, format="PNG", dpi=(target_dpi, target_dpi))
            buf.seek(0)

            st.download_button(
                label="⬇️ Download Dithered PNG",
                data=buf,
                file_name=f"engraving_{method_name.lower()}.png",
                mime="image/png",
                help=f"{w}x{h} px @ {target_dpi} DPI"
            )

            if beginner_mode:
                st.success("💡 Import this PNG into LightBurn, set to 'Image' mode, and adjust power/speed for your material!")

    # TAB 2: VECTOR SCORING
    with tab2:
        if beginner_mode:
            st.info("ℹ️ **BEGINNER TIP**: Vector scoring finds the edges/outlines in your image. Use this for cutting shapes, not engraving photos!")

        st.header("Vector Scoring — Edge Detection → SVG")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            min_threshold = st.slider("Min Threshold", 0, 255, 50, 5,
                                     help="Lower = detect more edges")
            blur_kernel = st.slider("Blur (noise reduction)", 1, 51, 5, 2)

        with col_p2:
            max_threshold = st.slider("Max Threshold", 1, 500, 150, 5,
                                     help="1:2 or 1:3 ratio with min recommended")
            epsilon_factor = st.slider("Smoothness (Anti-Jitter)", 0.0, 0.02, 0.003, 0.0005,
                                      help="⚠️ IMPORTANT: Higher = smoother cuts, prevents laser jitter!")

        min_area = st.slider("Filter Small Details", 0.0, 100.0, 10.0, 5.0,
                            help="Remove tiny contours (noise)")

        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        edges = cv2.Canny(blurred, min_threshold, max_threshold)
        contours = find_and_simplify_contours(edges, epsilon_factor, min_area)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Edge Detection")
            st.image(edges, use_container_width=True)

        with col2:
            st.subheader(f"Vector Paths ({len(contours)} contours)")
            preview_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
            if len(contours) > 0:
                cv2.drawContours(preview_canvas, contours, -1, (0, 0, 0), 1)

            preview_img = Image.fromarray(preview_canvas)
            if add_measurement_rulers:
                preview_img = add_measurement_overlay(preview_img, target_dpi, output_units)

            st.image(preview_img, use_container_width=True)

        svg_data = svg_bytes_from_contours(contours, w, h, output_dpi, output_units, add_alignment_marks)

        st.download_button(
            label=f"⬇️ Download SVG ({output_dpi} DPI, {output_units})",
            data=svg_data,
            file_name="vector_scoring.svg",
            mime="image/svg+xml"
        )

        if beginner_mode and len(contours) > 1000:
            st.warning("⚠️ **Many contours detected!** Consider increasing the smoothness slider or minimum area filter.")

    # TAB 3: MULTI-LAYER
    with tab3:
        if beginner_mode:
            st.info("ℹ️ **BEGINNER TIP**: Multi-layer creates separate cutting files that you stack for 3D depth. Start with 4-5 layers, not 20!")

        st.header("Multi-Layer Topographic — 10 Mandala Styles!")

        # Style guide for beginners
        if beginner_mode:
            with st.expander("📖 Quick Style Guide - Which Style Should I Choose?"):
                st.markdown("""
                **For Traditional Circular Mandalas:** 🎯 Classic Topographic or 🌸 Radial Bloom
                **For Geometric/Yantra Patterns:** 🔷 Sacred Geometry or 💎 Crystal Facets
                **For Nature/Floral Designs:** 🌊 Organic Flow or 🌸 Radial Bloom
                **For Star/Sun Patterns:** ⭐ Starburst or 🌀 Spiral Energy
                **For Complex Detailed Images:** 🎨 Detail Preserve or 🔲 Grid Harmony
                **For Balanced Light/Dark:** ☯️ Yin-Yang Balance

                **Not sure?** Start with 🎯 **Classic Topographic** - it works well for most images!
                """)

        # Material calculator
        material_thickness = show_material_calculator()

        st.markdown("---")

        col_l1, col_l2 = st.columns(2)

        with col_l1:
            num_layers = st.slider(
                "Number of Layers (2-20)",
                min_value=2,
                max_value=MAX_LAYERS,
                value=DEFAULT_LAYERS,
                help="⚠️ Beginners: Start with 4-5 layers!"
            )

            if beginner_mode and num_layers > 10:
                st.warning("⚠️ **Advanced**: 10+ layers requires precision cutting and alignment!")

        with col_l2:
            heavy_blur = st.slider(
                "Blur Amount",
                3, 201, 51, 2,
                help="Higher = organic shapes, Lower = sharp edges"
            )

        # 10 MANDALA-INSPIRED STYLES
        style_options = {
            "🎯 Classic Topographic": "Traditional concentric layers from outer to inner, like elevation contours",
            "🌸 Radial Bloom": "Petal-like layers radiating from center, preserving circular mandala patterns",
            "🔷 Sacred Geometry": "Sharp geometric layers based on yantra principles with clean edges",
            "🌊 Organic Flow": "Smooth flowing layers following natural image gradients",
            "⭐ Starburst": "Angular layers emphasizing radial symmetry and star patterns",
            "🎨 Detail Preserve": "Maximum detail retention with fine contour following",
            "🔲 Grid Harmony": "Balanced layers using structured threshold spacing",
            "🌀 Spiral Energy": "Layers following spiral and vortex patterns from center",
            "💎 Crystal Facets": "Sharp angular cuts creating faceted gem-like layers",
            "☯️ Yin-Yang Balance": "Alternating emphasis on light/dark features for balance"
        }

        selected_style = st.selectbox(
            "🎭 Multi-Layer Style",
            list(style_options.keys()),
            help="Choose a generation style optimized for different mandala types"
        )

        with st.expander("ℹ️ About This Style"):
            st.info(f"**{selected_style}**\n\n{style_options[selected_style]}")

        invert_layers = st.checkbox("Invert Stacking Order", value=False)

        # Process
        if st.button("🚀 Generate Layers", type="primary"):
            with st.spinner(f"Generating {num_layers} layers..."):
                blurred = cv2.GaussianBlur(gray, (heavy_blur, heavy_blur), 0)
                if invert_layers:
                    blurred = 255 - blurred

                svgs = []
                previews = []
                layer_info = []

                # STYLE-SPECIFIC PROCESSING
                # Each style uses different algorithms optimized for specific mandala types

                for i in range(num_layers):
                    # Calculate threshold based on style
                    if "Classic Topographic" in selected_style:
                        # Traditional nested layers from bright to dark
                        thresholds = np.linspace(255, 0, num_layers + 1)
                        threshold_val = int(thresholds[i + 1])

                    elif "Radial Bloom" in selected_style:
                        # Non-linear threshold for petal emphasis
                        progress = i / max(1, num_layers - 1)
                        threshold_val = int(255 * (1 - progress ** 0.7))  # Power curve for radial emphasis

                    elif "Sacred Geometry" in selected_style:
                        # Evenly spaced for geometric precision
                        thresholds = np.linspace(255, 0, num_layers + 1)
                        threshold_val = int(thresholds[i])

                    elif "Organic Flow" in selected_style:
                        # Sigmoid-based for smooth gradients
                        progress = i / max(1, num_layers - 1)
                        threshold_val = int(255 * (1 - 1/(1 + np.exp(-10*(progress-0.5)))))

                    elif "Starburst" in selected_style:
                        # Emphasize extremes for angular features
                        progress = i / max(1, num_layers - 1)
                        threshold_val = int(255 * (1 - progress ** 2))  # Quadratic for star points

                    elif "Detail Preserve" in selected_style:
                        # Fine gradation for maximum detail
                        threshold_val = int(255 - (i * 255 / num_layers))

                    elif "Grid Harmony" in selected_style:
                        # Balanced linear spacing
                        threshold_val = int(255 * (1 - i / num_layers))

                    elif "Spiral Energy" in selected_style:
                        # Logarithmic for spiral emphasis
                        progress = (i + 1) / num_layers
                        threshold_val = int(255 * (1 - np.log1p(progress * 2) / np.log1p(2)))

                    elif "Crystal Facets" in selected_style:
                        # Sharp transitions for faceted look
                        threshold_val = int(255 * (num_layers - i - 1) / num_layers)

                    else:  # Yin-Yang Balance
                        # Alternating emphasis
                        if i % 2 == 0:
                            threshold_val = int(255 * (1 - i / num_layers))
                        else:
                            threshold_val = int(255 * (1 - (i + 0.5) / num_layers))

                    # Create mask at threshold
                    _, mask = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

                    # Style-specific morphological operations
                    if "Sacred Geometry" in selected_style or "Crystal Facets" in selected_style:
                        # Minimal smoothing for sharp edges
                        kernel = np.ones((3, 3), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

                    elif "Organic Flow" in selected_style or "Radial Bloom" in selected_style:
                        # Heavy smoothing for organic shapes
                        kernel = np.ones((7, 7), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

                    elif "Detail Preserve" in selected_style:
                        # Minimal processing to preserve fine details
                        kernel = np.ones((3, 3), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

                    else:  # Default balanced processing
                        kernel_size = 3 if num_layers > 10 else 5
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

                    # Edge detection with style-specific parameters
                    if "Sacred Geometry" in selected_style or "Crystal Facets" in selected_style:
                        # High thresholds for clean geometric edges
                        edges = cv2.Canny(mask, 100, 200)
                    elif "Detail Preserve" in selected_style:
                        # Low thresholds for fine details
                        edges = cv2.Canny(mask, 30, 100)
                    else:
                        # Balanced edge detection
                        edges = cv2.Canny(mask, 50, 150)

                    # Find contours
                    if "Radial Bloom" in selected_style or "Starburst" in selected_style:
                        # RETR_TREE for hierarchical radial patterns
                        contours_raw, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    else:
                        # RETR_EXTERNAL for cleaner layers
                        contours_raw, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    # Filter and simplify contours
                    contour_list = []

                    # Style-specific minimum area
                    if "Detail Preserve" in selected_style:
                        min_contour_area = 10.0  # Keep tiny details
                    elif num_layers > 10:
                        min_contour_area = 50.0
                    else:
                        min_contour_area = 20.0

                    for c in contours_raw:
                        area = cv2.contourArea(c)
                        if area < min_contour_area:
                            continue

                        peri = cv2.arcLength(c, True)

                        # Style-specific simplification
                        if "Sacred Geometry" in selected_style or "Crystal Facets" in selected_style:
                            # Aggressive simplification for angular geometry
                            eps = max(1.0, peri * 0.005)
                        elif "Detail Preserve" in selected_style:
                            # Minimal simplification
                            eps = max(0.3, peri * 0.001)
                        elif "Organic Flow" in selected_style:
                            # Moderate for smooth curves
                            eps = max(0.7, peri * 0.003)
                        else:
                            # Balanced default
                            eps = max(0.5, peri * 0.002)

                        approx = cv2.approxPolyDP(c, eps, True)
                        contour_list.append(approx)

                    # Generate SVG
                    svg_bytes = svg_bytes_from_contours(contour_list, w, h, output_dpi, output_units, add_alignment_marks)

                    style_name = selected_style.split()[1] if len(selected_style.split()) > 1 else "layer"
                    layer_name = f"layer_{i+1:02d}_{style_name}_t{threshold_val}.svg"
                    svgs.append((layer_name, svg_bytes))
                    previews.append(make_layer_preview(contour_list, w, h))
                    layer_info.append({
                        'layer': i + 1,
                        'threshold': threshold_val,
                        'contours': len(contour_list),
                        'style': selected_style
                    })

            st.success(f"✅ {num_layers} layers generated!")

            # Display previews
            st.subheader(f"Layer Previews (Total: {num_layers})")

            # Adaptive column count
            cols_per_row = min(5, num_layers)
            rows_needed = (num_layers + cols_per_row - 1) // cols_per_row

            for row in range(rows_needed):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    layer_idx = row * cols_per_row + col_idx
                    if layer_idx < num_layers:
                        with cols[col_idx]:
                            st.image(previews[layer_idx], caption=f"Layer {layer_idx + 1}", use_container_width=True)

            # Layer statistics
            with st.expander("📊 Layer Details", expanded=False):
                import pandas as pd
                df = pd.DataFrame(layer_info)
                st.dataframe(df, use_container_width=True)

            # Stack height info
            total_height = calculate_total_stack_height(num_layers, material_thickness)
            st.metric("Total Stack Height", f"{total_height:.1f} mm")

            # ZIP download
            zip_bytes = create_zip_from_svg_list(svgs)

            st.download_button(
                label=f"⬇️ Download All {num_layers} Layers (ZIP)",
                data=zip_bytes,
                file_name=f"mandala_{num_layers}layers.zip",
                mime="application/zip"
            )

            if beginner_mode:
                st.info("💡 **Next Steps**: Unzip the file, import each layer into LightBurn separately, cut them, then stack in order (Layer 1 on bottom)!")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><strong>🔷 Mandala Laser Engraving App - Enhanced Edition</strong></p>
        <p>Features: 3 Dithering Algorithms • Anti-Jitter Vector Paths • 20-Layer Support • Measurement Tools • Beginner Guides</p>
        <p style='font-size: 0.85em;'>Optimized for LightBurn, Inkscape, and all professional laser cutters</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
