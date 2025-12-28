import io
import zipfile
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
import svgwrite
from PIL import Image


def load_image(uploaded) -> Image.Image:
    image = Image.open(uploaded).convert("RGB")
    return image


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    # RGB to BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    # BGR to RGB
    arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)


def floyd_steinberg_dither(gray: np.ndarray) -> np.ndarray:
    # Input: 2D uint8 array 0..255
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
    out = np.clip(img, 0, 255).astype(np.uint8)
    return out


def contours_to_svg(contours: List[np.ndarray], width: int, height: int, stroke_width=1) -> str:
    # Create SVG with proper namespace for LightBurn/Inkscape compatibility
    dwg = svgwrite.Drawing(
        size=(f"{width}px", f"{height}px"),
        viewBox=f"0 0 {width} {height}",
        profile='full'
    )
    # Add namespace for better compatibility
    dwg.attribs['xmlns'] = 'http://www.w3.org/2000/svg'
    dwg.attribs['xmlns:xlink'] = 'http://www.w3.org/1999/xlink'

    for cnt in contours:
        if len(cnt) < 2:
            continue
        pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
        # Create path data
        path = "M {} {}".format(pts[0][0], pts[0][1])
        for (x, y) in pts[1:]:
            path += " L {} {}".format(x, y)
        path += " Z"
        dwg.add(dwg.path(d=path, fill="none", stroke="black", stroke_width=stroke_width))
    return dwg.tostring()


def find_and_simplify_contours(binary: np.ndarray, epsilon_factor=0.002) -> List[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    simplified = []
    for c in contours:
        if cv2.contourArea(c) < 1.0:
            continue
        peri = cv2.arcLength(c, True)
        eps = max(1.0, peri * epsilon_factor)
        approx = cv2.approxPolyDP(c, eps, True)
        simplified.append(approx)
    return simplified


def svg_bytes_from_contours(contours: List[np.ndarray], w: int, h: int) -> bytes:
    svg_text = contours_to_svg(contours, w, h)
    return svg_text.encode("utf-8")


def make_layer_preview(contours: List[np.ndarray], w: int, h: int) -> Image.Image:
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.drawContours(canvas, contours, -1, (0, 0, 0), 1)
    return Image.fromarray(canvas)


def create_zip_from_svg_list(svgs: List[Tuple[str, bytes]]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in svgs:
            zf.writestr(name, data)
    mem.seek(0)
    return mem.read()


st.set_page_config(
    page_title="Mandala Laser Engraving App",
    page_icon="🔷",
    layout="wide"
)

st.title("🔷 Mandala Laser Engraving App")
st.markdown("Convert images into laser-engraving optimized formats with up to 10 layers for stunning 3D mandala effects")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

tab1, tab2, tab3 = st.tabs(["Photo Engraving (Dither)", "Vector Scoring (Edges)", "Multi-Layer Topographic"])

if uploaded is None:
    st.info("Upload an image to begin. Use the tabs to choose a processing mode.")

if uploaded:
    image = load_image(uploaded)
    w, h = image.size
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    with tab1:
        st.header("Photo Engraving — Floyd–Steinberg Dithering")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        with col2:
            dithered = floyd_steinberg_dither(gray)
            pil_dither = Image.fromarray(dithered).convert("L")
            st.image(pil_dither, caption="Dithered (1-bit) preview", clamp=True)
            buf = io.BytesIO()
            pil_dither.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download Dithered PNG", data=buf, file_name="dithered.png", mime="image/png")

    with tab2:
        st.header("Vector Scoring — Edge Detection → SVG")
        min_t = st.slider("Canny min threshold", 0, 255, 50)
        max_t = st.slider("Canny max threshold", 1, 500, 150)
        blur_k = st.slider("Gaussian blur kernel (odd)", 1, 51, 5, step=2)
        eps_factor = st.slider("Contour simplify epsilon factor", 0.0, 0.02, 0.002, step=0.0005)

        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(blurred, min_t, max_t)
        st.subheader("Edge preview")
        st.image(edges, caption="Canny edges (raster)", use_column_width=True)

        binary = edges.copy()
        contours = find_and_simplify_contours(binary, epsilon_factor=eps_factor)
        st.write(f"Found {len(contours)} simplified contours")

        svg_data = svg_bytes_from_contours(contours, w, h)
        st.download_button("Download SVG", data=svg_data, file_name="edges.svg", mime="image/svg+xml")

        # show vector preview as raster of contours
        preview_canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
        cv2.drawContours(preview_canvas, contours, -1, (0, 0, 0), 1)
        st.image(preview_canvas, caption="Vectorized contours (preview)")

    with tab3:
        st.header("Multi-Layer Topographic — Stacked 3D Mandala Layers")
        st.markdown("Create stacked layers where darker areas are cut separately for 3D depth effect")
        layers = st.slider("Number of layers (2-10)", 2, 10, 4)
        heavy_blur = st.slider("Blur kernel (odd, larger = more organic)", 3, 201, 51, step=2)

        blurred = cv2.GaussianBlur(gray, (heavy_blur, heavy_blur), 0)
        st.subheader("Blurred base preview")
        st.image(blurred, clamp=True)

        # compute thresholds dividing brightness into ranges
        thresholds = np.linspace(0, 255, layers + 1)
        svgs = []
        previews = []
        for i in range(layers):
            lo = thresholds[i]
            hi = thresholds[i + 1]
            # for darker layers first (lo..hi). We will treat layer 1 as darkest band
            mask = cv2.inRange(blurred, int(lo), int(hi))
            # remove tiny noise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contour_list = find_and_simplify_contours(mask, epsilon_factor=0.002)
            svg_bytes = svg_bytes_from_contours(contour_list, w, h)
            name = f"layer_{i+1}.svg"
            svgs.append((name, svg_bytes))
            previews.append(make_layer_preview(contour_list, w, h))

        st.subheader("Layer previews")
        cols = st.columns(min(4, layers))
        for idx, p in enumerate(previews):
            c = cols[idx % len(cols)]
            c.image(p, caption=f"Layer {idx+1}")

        zip_bytes = create_zip_from_svg_list(svgs)
        st.download_button("Download ZIP of SVG layers", data=zip_bytes, file_name="layers.zip", mime="application/zip")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p><strong>Mandala Laser Engraving App</strong> | Optimized for LightBurn, Inkscape, and laser cutters</p>
        <p>Features: Floyd-Steinberg Dithering • Canny Edge Detection • Multi-Layer 3D Stacking</p>
        </div>
        """,
        unsafe_allow_html=True
    )
