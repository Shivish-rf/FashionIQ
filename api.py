from contextlib import asynccontextmanager
from typing import Annotated, Literal
import os
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Reuse your core math, color, sizing, and style functions
from style_and_size_app import (
    analyze_skin_tone,
    classify_body_shape,
    ellipse_circumference,
    get_cm_per_pixel,
    get_measurement_coords,
    map_clothing_size,
    measure_segment_width_px,
)

MODEL_PATH = "pose_landmarker_heavy.task"
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB limit per image
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the MediaPipe PoseLandmarker once on startup and close it on shutdown."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        output_segmentation_masks=True,
    )
    ml_models["pose_detector"] = vision.PoseLandmarker.create_from_options(options)
    yield
    ml_models["pose_detector"].close()
    ml_models.clear()


app = FastAPI(title="Virtual Sizing & Style API", lifespan=lifespan)

# Allow mobile/web frontends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend domain in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


async def decode_upload_in_ram(upload: UploadFile) -> np.ndarray:
    """Validate and decode an uploaded image strictly in memory."""
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{upload.content_type}'. Use JPEG, PNG, or WebP.",
        )

    raw_bytes = await upload.read()
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 5 MB size limit.",
        )

    np_arr = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Corrupted image: {upload.filename}")
    return image


def process_bgr_image(image: np.ndarray, detector):
    """Resize, run PoseLandmarker, and return image, landmarks, and cleaned binary mask."""
    h, w = image.shape[:2]
    scale = 1024 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, (new_w, new_h), interpolation=interp)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if not results.pose_landmarks or not results.segmentation_masks:
        raise HTTPException(
            status_code=422,
            detail="Could not detect a full person. Ensure head-to-feet visibility and good lighting.",
        )

    landmarks = results.pose_landmarks[0]
    raw_mask = np.squeeze(results.segmentation_masks[0].numpy_view())
    mask = (cv2.resize(raw_mask, (new_w, new_h)) > 0.5).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return image, landmarks, mask, new_h, new_w


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": "pose_detector" in ml_models}


@app.post("/analyze")
async def analyze_body(
    front_image: Annotated[UploadFile, File(description="Full-body front photo in A-pose")],
    side_image: Annotated[UploadFile, File(description="Full-body 90-degree side photo")],
    height_cm: Annotated[float, Form(ge=100.0, le=250.0)],
    weight_kg: Annotated[float, Form(ge=30.0, le=250.0)],
    gender: Annotated[Literal["male", "female"], Form()] = "male",
):
    front_bgr = await decode_upload_in_ram(front_image)
    side_bgr = await decode_upload_in_ram(side_image)

    detector = ml_models["pose_detector"]

    try:
        f_img, f_lm, f_mask, f_h, f_w = process_bgr_image(front_bgr, detector)
        _, s_lm, s_mask, s_h, s_w = process_bgr_image(side_bgr, detector)

        f_scale = get_cm_per_pixel(f_mask, height_cm)
        s_scale = get_cm_per_pixel(s_mask, height_cm)

        f_coords = get_measurement_coords(f_lm, f_h, f_w)
        s_coords = get_measurement_coords(s_lm, s_h, s_w)

        sh_x, sh_y = f_coords["shoulder"]
        shoulder_width_cm = measure_segment_width_px(f_mask, sh_y, sh_x) * f_scale

        measurements = {}
        for part in ["chest", "waist", "hip", "thigh"]:
            fx, fy = f_coords[part]
            sx, sy = s_coords[part]
            width_cm = measure_segment_width_px(f_mask, fy, fx) * f_scale
            depth_cm = measure_segment_width_px(s_mask, sy, sx) * s_scale
            circ_cm = ellipse_circumference(width_cm, depth_cm)

            measurements[part] = {
                "width_cm": round(width_cm, 1),
                "depth_cm": round(depth_cm, 1),
                "circumference_cm": round(circ_cm, 1),
                "circumference_in": round(circ_cm / 2.54, 1),
            }

        bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 1)
        color_profile = analyze_skin_tone(f_img, f_lm)
        size_profile = map_clothing_size(measurements, gender=gender)
        style_profile = classify_body_shape(measurements, shoulder_width_cm)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    return {
        "bmi": bmi,
        "measurements": measurements,
        "sizing": size_profile,
        "style": style_profile,
        "color_analysis": color_profile,
    }