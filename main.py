import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import streamlit as st

from style_and_size_app import (
    analyze_skin_tone,
    apply_weight_calibration,
    classify_body_shape,
    ellipse_circumference,
    get_cm_per_pixel,
    get_measurement_coords,
    map_clothing_size,
    measure_segment_width_px,
    validate_pose_and_angle,
)

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

COLOR_HEX_MAP = {
    "Coral": "#FF7F50",
    "Warm Peach": "#FFDAB9",
    "Turquoise": "#40E0D0",
    "Cream": "#FFFDD0",
    "Camel": "#C19A6B",
    "Mint Green": "#98FF98",
    "Stark Black": "#000000",
    "Icy Blue": "#D6F1FF",
    "Dark Charcoal": "#36454F",
    "Olive Green": "#556B2F",
    "Mustard": "#E1AD01",
    "Rust": "#B7410E",
    "Terracotta": "#E2725B",
    "Warm Brown": "#8B5A2B",
    "Teal": "#008080",
    "Pastel Pink": "#FFD1DC",
    "Neon Blue": "#1F51FF",
    "Pure White": "#FFFFFF",
    "Lavender": "#B57EDC",
    "Powder Blue": "#B0E0E6",
    "Dusty Rose": "#DCAE96",
    "Soft Navy": "#2C3E50",
    "Cool Gray": "#8C92AC",
    "Orange": "#FFA500",
    "Mustard Yellow": "#E1AD01",
    "Emerald Green": "#50C878",
    "Royal Blue": "#4169E1",
    "Burgundy": "#800020",
    "True Black": "#111111",
    "Beige": "#F5F5DC",
    "Earthy Olive": "#6B8E23",
    "Jade Green": "#00A86B",
    "Dusty Pink": "#D8A1A6",
    "Soft White": "#F8F8FF",
    "Navy": "#000080",
    "Taupe": "#483C32",
    "Overly Neon Yellow": "#DFFF00",
    "Harsh Fluorescent Orange": "#FF5F1F",
}

st.set_page_config(
    page_title="AI Virtual Sizing & Stylist",
    page_icon="👕",
    layout="wide",
)


@st.cache_resource
def load_pose_detector():
    """Download the MediaPipe model if missing, then load and cache it."""
    if not os.path.exists(MODEL_PATH):
        tmp_path = f"{MODEL_PATH}.tmp"
        with st.spinner("Downloading MediaPipe Pose model (~30 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, tmp_path)
            os.replace(tmp_path, MODEL_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        output_segmentation_masks=True,
    )
    return vision.PoseLandmarker.create_from_options(options)


def decode_uploaded_file(uploaded_file) -> np.ndarray:
    """Decode a Streamlit UploadedFile or camera_input directly in RAM."""
    raw_bytes = uploaded_file.getvalue()
    np_arr = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Please try another photo.")
    return image


def process_bgr_image(image: np.ndarray, detector):
    """Resize image, run PoseLandmarker, and return image, landmarks, and binary mask."""
    h, w = image.shape[:2]
    scale = 1024 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, (new_w, new_h), interpolation=interp)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if not results.pose_landmarks or not results.segmentation_masks:
        raise ValueError(
            "Could not detect a full person. Make sure your full body (head to feet) is visible."
        )

    landmarks = results.pose_landmarks[0]
    raw_mask = np.squeeze(results.segmentation_masks[0].numpy_view())
    mask = (cv2.resize(raw_mask, (new_w, new_h)) > 0.5).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return image, landmarks, mask, new_h, new_w


def render_color_swatches(colors):
    """Render visual color chips using HTML/CSS."""
    chips_html = '<div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:8px;">'
    for name in colors:
        hex_code = COLOR_HEX_MAP.get(name, "#CCCCCC")
        chips_html += (
            f'<div style="display:flex; align-items:center; gap:8px; '
            f'padding:6px 12px; border-radius:20px; border:1px solid #ddd; background:#f9f9f9;">'
            f'<span style="width:16px; height:16px; border-radius:50%; '
            f'background:{hex_code}; border:1px solid #999; display:inline-block;"></span>'
            f'<span style="font-size:14px; color:#222;">{name}</span>'
            f"</div>"
        )
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)


def capture_guided_photo(view_name, input_mode, detector):
    """
    Guided camera/upload widget with instant pose & angle validation.
    Returns (bgr_image, landmarks, mask, h, w) if captured, else None.
    """
    is_front = view_name == "front"
    title = "1. Front View (A-Pose)" if is_front else "2. Side View (90° Profile)"
    instructions = (
        "Face straight ahead, stand ~2 meters back, and hold arms slightly away from your waist."
        if is_front
        else "Turn 90° sideways, stand straight, and keep arms relaxed at your sides."
    )

    st.subheader(title)
    st.caption(instructions)

    if input_mode == "Take Photos with Camera":
        file_obj = st.camera_input(
            f"Click {view_name.capitalize()} Photo",
            key=f"{view_name}_cam",
        )
    else:
        file_obj = st.file_uploader(
            f"Upload {view_name.capitalize()} Photo",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"{view_name}_up",
        )
        if file_obj:
            st.image(
                file_obj,
                caption=f"{view_name.capitalize()} Preview",
                use_container_width=True,
            )

    if not file_obj:
        return None

    try:
        bgr = decode_uploaded_file(file_obj)
        img, lm, mask, h, w = process_bgr_image(bgr, detector)
        is_valid, warnings = validate_pose_and_angle(lm, view=view_name)

        if is_valid:
            st.success(f"{view_name.capitalize()} photo looks great!")
        else:
            for warn in warnings:
                st.warning(warn)

        return (img, lm, mask, h, w)
    except Exception as e:
        st.error(f"Photo check failed: {e}")
        return None


# =========================================
# UI LAYOUT
# =========================================

st.title("👕 AI Virtual Sizing, Body Shape & Color Stylist")
st.write(
    "Provide a **Front** photo (A-pose) and a **Side** photo (90° profile) along with your "
    "height and weight to estimate your body measurements, clothing size, flattering styles, "
    "and seasonal color palette."
)

# Load model once at startup (auto-downloads if missing)
try:
    detector = load_pose_detector()
except Exception as e:
    st.error(f"Failed to load MediaPipe model: {e}")
    st.stop()

# Sidebar: User details & capture mode
with st.sidebar:
    st.header("Your Details")
    height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0, step=0.5)
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=72.0, step=0.5)
    gender = st.selectbox("Gender (for sizing chart)", ["male", "female"])

    st.divider()
    input_mode = st.radio("Photo Input Method", ["Upload Photos", "Take Photos with Camera"])

    st.info(
        "**Tips for best accuracy:**\n"
        "- Wear fitted clothes\n"
        "- Stand ~2 meters away (head to feet visible)\n"
        "- Front view: hold arms slightly away from waist (A-pose)\n"
        "- Side view: turn a full 90° sideways"
    )

# Main input area: Guided Front & Side photos
col1, col2 = st.columns(2)
with col1:
    front_data = capture_guided_photo("front", input_mode, detector)
with col2:
    side_data = capture_guided_photo("side", input_mode, detector)

st.divider()

# Run Analysis Button
if st.button("Analyze Size, Style & Colors", type="primary", use_container_width=True):
    if not front_data or not side_data:
        st.warning("Please capture or upload both a valid Front and Side photo first.")
    else:
        with st.spinner("Calibrating measurements with your height & weight..."):
            try:
                f_img, f_lm, f_mask, f_h, f_w = front_data
                _, s_lm, s_mask, s_h, s_w = side_data

                f_scale = get_cm_per_pixel(f_mask, height_cm)
                s_scale = get_cm_per_pixel(s_mask, height_cm)

                f_coords = get_measurement_coords(f_lm, f_h, f_w)
                s_coords = get_measurement_coords(s_lm, s_h, s_w)

                sh_x, sh_y = f_coords["shoulder"]
                shoulder_width_cm = measure_segment_width_px(f_mask, sh_y, sh_x) * f_scale

                raw_measurements = {}
                for part in ["chest", "waist", "hip", "thigh"]:
                    fx, fy = f_coords[part]
                    sx, sy = s_coords[part]
                    width_cm = measure_segment_width_px(f_mask, fy, fx) * f_scale
                    depth_cm = measure_segment_width_px(s_mask, sy, sx) * s_scale
                    circ_cm = ellipse_circumference(width_cm, depth_cm)

                    raw_measurements[part] = {
                        "width_cm": round(width_cm, 1),
                        "depth_cm": round(depth_cm, 1),
                        "circumference_cm": round(circ_cm, 1),
                        "circumference_in": round(circ_cm / 2.54, 1),
                    }

                # Apply height + weight calibration & baggy-clothing sanity check
                measurements, calibration_alerts = apply_weight_calibration(
                    raw_measurements, height_cm, weight_kg, gender=gender
                )

                bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 1)
                color_profile = analyze_skin_tone(f_img, f_lm)
                size_profile = map_clothing_size(measurements, gender=gender)
                style_profile = classify_body_shape(measurements, shoulder_width_cm)

                # =========================================
                # DISPLAY RESULTS
                # =========================================
                if calibration_alerts:
                    for alert in calibration_alerts:
                        st.warning(alert)
                else:
                    st.success("Analysis complete — measurements verified against your height & weight!")

                # 1. Top Summary Cards
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Top Size", size_profile["top_size"])
                m2.metric("Bottom Size", size_profile["bottom_size"])
                m3.metric("Body Shape", style_profile["body_shape"])
                m4.metric("BMI", f"{bmi}")

                st.divider()

                # 2. Detailed Tabs
                tab_meas, tab_style, tab_color = st.tabs(
                    ["📏 Body Measurements", "🧥 Style & Fit Guide", "🎨 Skin Tone & Colors"]
                )

                with tab_meas:
                    st.subheader("Calibrated Circumferences")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    for col, part in zip([mc1, mc2, mc3, mc4], ["chest", "waist", "hip", "thigh"]):
                        d = measurements[part]
                        col.metric(
                            label=part.capitalize(),
                            value=f"{d['circumference_cm']} cm",
                            delta=f"{d['circumference_in']} in",
                            delta_color="off",
                        )

                    with st.expander("View Raw vs. Calibrated Breakdown"):
                        st.table(
                            [
                                {
                                    "Body Part": p.capitalize(),
                                    "Front Width (cm)": d["width_cm"],
                                    "Side Depth (cm)": d["depth_cm"],
                                    "Raw Circ (cm)": d.get("raw_circumference_cm", d["circumference_cm"]),
                                    "Calibrated Circ (cm)": d["circumference_cm"],
                                    "Calibrated Circ (in)": d["circumference_in"],
                                }
                                for p, d in measurements.items()
                            ]
                        )

                with tab_style:
                    st.subheader(f"Body Shape: {style_profile['body_shape']}")
                    st.markdown("**Recommended Cuts & Silhouettes:**")
                    for tip in style_profile["recommended_styles"]:
                        st.markdown(f"- {tip}")

                    st.markdown("**Proportion & Fit Notes:**")
                    for note in size_profile["fit_notes"]:
                        st.markdown(f"- {note}")

                with tab_color:
                    if "error" in color_profile:
                        st.error(color_profile["error"])
                    else:
                        st.subheader(f"Season: {color_profile['season']}")
                        st.write(
                            f"**Skin Depth:** {color_profile['depth']}  |  "
                            f"**Undertone:** {color_profile['undertone']}"
                        )

                        st.markdown("**Colors That Suit You Best:**")
                        render_color_swatches(color_profile["best_colors"])

                        st.markdown("<br>**Colors to Avoid:**", unsafe_allow_html=True)
                        render_color_swatches(color_profile["avoid_colors"])

            except Exception as e:
                st.error(f"Analysis failed: {e}")