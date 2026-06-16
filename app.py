"""
app.py — Bangla Handwritten Word Recognition (Streamlit UI)
Draw a Bangla word on the canvas → each character is segmented & predicted.
"""

import os
import json
import pathlib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bangla OCR — BanglaLekha",
    page_icon="🖊️",
    layout="wide",
)

# ── Custom CSS (premium dark theme) ──────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+Bengali:wght@400;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e8e8f0;
    }

    /* Header */
    .ocr-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .ocr-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: .3rem;
    }
    .ocr-header p {
        color: #a0a0c0;
        font-size: 1.05rem;
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* Result word */
    .bangla-word {
        font-family: 'Noto Serif Bengali', serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #f9a8d4, #c084fc, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: .1em;
        padding: 1rem;
    }

    /* Per-character chips */
    .char-chip {
        display: inline-block;
        background: rgba(96,165,250,0.15);
        border: 1px solid rgba(96,165,250,0.35);
        border-radius: 12px;
        padding: .4rem 1rem;
        margin: .3rem;
        font-family: 'Noto Serif Bengali', serif;
        font-size: 1.4rem;
        color: #bfdbfe;
        transition: all .2s;
    }

    /* Confidence bar */
    .conf-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        height: 10px;
        margin-top: 4px;
    }
    .conf-bar-fill {
        height: 10px;
        border-radius: 6px;
        background: linear-gradient(90deg, #34d399, #60a5fa);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        color: white;
        border: none;
        border-radius: 12px;
        padding: .6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: opacity .2s, transform .1s;
        width: 100%;
    }
    .stButton > button:hover { opacity: .88; transform: translateY(-1px); }

    /* Metric boxes */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15,12,41,0.8) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    hr { border-color: rgba(255,255,255,0.12); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Lazy imports so Streamlit starts fast ────────────────────────────────────
@st.cache_resource(show_spinner="Loading TensorFlow model…")
def load_model_and_labels():
    import tensorflow as tf

    model_path  = pathlib.Path("models/model.keras")
    labels_path = pathlib.Path("labels.json")

    if not model_path.exists():
        st.error("❌  `models/model.keras` not found. Please run `python train.py` first.")
        st.stop()
    if not labels_path.exists():
        st.error("❌  `labels.json` not found. Please run `python train.py` first.")
        st.stop()

    model = tf.keras.models.load_model(str(model_path))
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)   # {str_index: bangla_char}

    return model, labels


# ── Character segmentation ────────────────────────────────────────────────────
def segment_characters(canvas_array: np.ndarray, min_area: int = 80):
    """
    Given an RGBA canvas image (numpy uint8), return a list of
    cropped grayscale character images sorted left-to-right.
    """
    import cv2
    
    # Convert RGBA to grayscale with a solid white background using alpha compositing
    rgba_img = Image.fromarray(canvas_array)
    white_bg = Image.new("RGBA", rgba_img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, rgba_img).convert("L")
    gray = np.array(composite)

    # Invert so characters are white on black (threshold expects that)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close to connect broken strokes
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        bboxes.append((x, y, w, h))

    # Sort left-to-right
    bboxes.sort(key=lambda b: b[0])

    # Crop each character; pad to square; resize to 64×64
    chars = []
    for (x, y, w, h) in bboxes:
        roi = gray[y: y + h, x: x + w]
        
        # Perfect square centering padding (handles w != h accurately)
        sz = max(w, h) + 16
        square = Image.new("L", (sz, sz), 255)
        x_offset = (sz - w) // 2
        y_offset = (sz - h) // 2
        square.paste(Image.fromarray(roi), (x_offset, y_offset))
        
        pad = square.resize((64, 64), Image.Resampling.LANCZOS)
        chars.append((np.array(pad), (x, y, w, h)))

    return chars


def preprocess_char(char_img: np.ndarray) -> np.ndarray:
    """Normalise a 64×64 grayscale char for model input."""
    arr = char_img.astype("float32") / 255.0
    arr = 1.0 - arr            # invert: dark char on white bg → white char on black
    arr = arr[..., np.newaxis]  # add channel dim
    return arr[np.newaxis]      # batch dim


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    stroke_width = st.slider("Brush Size",  min_value=3, max_value=20, value=8)
    stroke_color = st.color_picker("Stroke Color", "#000000")
    bg_color     = st.color_picker("Canvas Background", "#FFFFFF")
    min_area     = st.slider("Min Contour Area", 20, 300, 80,
                              help="Ignore tiny noise specks smaller than this.")
    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown(
        """
1. Draw **one Bangla word** on the canvas.
2. Press **Recognize Word**.
3. The app segments each character and predicts it.
4. The assembled Bangla word appears below.
        """
    )
    st.markdown("---")
    st.markdown("*BanglaLekha-Isolated · SICIP @ BRAC University*")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ocr-header">
        <h1>🖊️ Bangla Handwritten Word Recognition</h1>
        <p>Draw a Bangla word on the canvas below and click <strong>Recognize Word</strong>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Load model ────────────────────────────────────────────────────────────────
model, labels = load_model_and_labels()

# ── Canvas ────────────────────────────────────────────────────────────────────
try:
    from streamlit_drawable_canvas import st_canvas

    col_canvas, col_result = st.columns([3, 2])

    with col_canvas:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("✏️ Drawing Canvas")

        canvas_result = st_canvas(
            fill_color   = "rgba(255,255,255,0)",
            stroke_width = stroke_width,
            stroke_color = stroke_color,
            background_color = bg_color,
            height       = 300,
            width        = 700,
            drawing_mode = "freedraw",
            key          = "canvas",
            display_toolbar = True,
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            recognize = st.button("🔍 Recognize Word", type="primary")
        with btn_col2:
            clear     = st.button("🗑️ Clear")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Recognition ─────────────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Recognition Result")

        if recognize and canvas_result.image_data is not None:
            canvas_arr = canvas_result.image_data.astype("uint8")

            # Check canvas is not blank
            alpha = canvas_arr[:, :, 3]
            if alpha.max() == 0:
                st.warning("Canvas is empty. Please draw something first.")
            else:
                chars_data = segment_characters(canvas_arr, min_area=min_area)

                if not chars_data:
                    st.warning("No characters detected. Try drawing larger strokes.")
                else:
                    char_predictions = []
                    for char_img, bbox in chars_data:
                        inp   = preprocess_char(char_img)
                        probs = model.predict(inp, verbose=0)[0]
                        top1  = int(np.argmax(probs))
                        conf  = float(probs[top1])
                        char  = labels.get(str(top1), "?")
                        char_predictions.append({
                            "char":  char,
                            "conf":  conf,
                            "probs": probs,
                            "bbox":  bbox,
                        })

                    recognized_word = "".join(p["char"] for p in char_predictions)

                    # ── Result word ──────────────────────────────────────────
                    st.markdown(
                        f'<div class="bangla-word">{recognized_word}</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("---")

                    # ── Per-character breakdown ──────────────────────────────
                    st.markdown("**Per-character predictions:**")
                    chips = "".join(
                        f'<span class="char-chip">{p["char"]}</span>'
                        for p in char_predictions
                    )
                    st.markdown(chips, unsafe_allow_html=True)

                    st.markdown("")

                    # ── Confidence scores ────────────────────────────────────
                    for i, pred in enumerate(char_predictions):
                        conf_pct = pred["conf"] * 100
                        fill_w   = int(conf_pct)
                        st.markdown(
                            f"""
                            <div style="margin:.4rem 0;">
                              <span style="font-family:'Noto Serif Bengali',serif;
                                           font-size:1.1rem;color:#e8e8f0;">
                                {pred['char']}
                              </span>
                              <span style="color:#a0a0c0;font-size:.85rem;">
                                &nbsp;{conf_pct:.1f}%
                              </span>
                              <div class="conf-bar-bg">
                                <div class="conf-bar-fill" style="width:{fill_w}%;"></div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # ── Top-3 for each char ──────────────────────────────────
                    with st.expander("🔬 Top-3 Predictions Per Character"):
                        for i, pred in enumerate(char_predictions):
                            top3_idx  = np.argsort(pred["probs"])[::-1][:3]
                            top3_data = [
                                (labels.get(str(idx), "?"), pred["probs"][idx])
                                for idx in top3_idx
                            ]
                            st.markdown(f"**Character {i+1}:**")
                            for rank, (ch, prob) in enumerate(top3_data, 1):
                                st.markdown(
                                    f"&nbsp;&nbsp;{rank}. "
                                    f"<span style='font-family:Noto Serif Bengali,serif;"
                                    f"font-size:1.2rem;'>{ch}</span> — "
                                    f"`{prob*100:.1f}%`",
                                    unsafe_allow_html=True,
                                )

        elif not recognize:
            st.info("👈 Draw on the canvas and click **Recognize Word**.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Model info ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("ℹ️ Model Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input Size",  "64 × 64")
    c2.metric("Channels",    "Grayscale (1)")
    c3.metric("Classes",     f"{len(labels)}")
    c4.metric("Framework",   "TensorFlow/Keras")

except ImportError:
    st.error(
        "❌ `streamlit-drawable-canvas` is not installed. "
        "Run `pip install streamlit-drawable-canvas` and restart."
    )
