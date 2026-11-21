import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# BlazePose landmark indices
LM = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
}


def analyze_image(image_path, detector):
    """Run PoseLandmarker and return landmarks, binary mask, and image dimensions."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Resize while preserving aspect ratio (max side 1024)
    h, w = image.shape[:2]
    scale = 1024 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    image = cv2.resize(image, (new_w, new_h), interpolation=interp)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if not results.pose_landmarks or not results.segmentation_masks:
        raise ValueError(f"Could not detect person/mask in {image_path}")

    landmarks = results.pose_landmarks[0]
    raw_mask = np.squeeze(results.segmentation_masks[0].numpy_view())
    mask = (cv2.resize(raw_mask, (new_w, new_h)) > 0.5).astype(np.uint8)

    # Clean small mask noise with morphological opening/closing
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return landmarks, mask, new_h, new_w


def get_cm_per_pixel(mask, user_height_cm):
    """Calibrate pixel-to-cm ratio using the full vertical span of the silhouette."""
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        raise ValueError("Empty segmentation mask")
    pixel_height = int(ys.max() - ys.min())
    return user_height_cm / pixel_height


def measure_segment_width_px(mask, y_center, x_ref, band=3):
    """
    Measure the horizontal pixel width of the contiguous silhouette segment
    containing x_ref. Using the contiguous segment prevents A-pose arms from
    being counted in chest/waist width.
    """
    h, w = mask.shape
    widths = []

    for y in range(max(0, y_center - band), min(h, y_center + band + 1)):
        row = mask[y, :]
        x = int(np.clip(x_ref, 0, w - 1))

        # If x_ref lands slightly outside the mask, search nearby for the closest pixel
        if row[x] == 0:
            fg = np.where(row > 0)[0]
            if len(fg) == 0:
                continue
            x = int(fg[np.argmin(np.abs(fg - x))])

        left = x
        while left > 0 and row[left - 1] > 0:
            left -= 1

        right = x
        while right < w - 1 and row[right + 1] > 0:
            right += 1

        widths.append(right - left + 1)

    if not widths:
        raise ValueError(f"No foreground pixels found near y={y_center}")
    return float(np.median(widths))


def get_measurement_coords(landmarks, height, width):
    """Compute (x_ref, y_level) for chest, waist, hip, and left thigh."""
    def pt(name):
        lm = landmarks[LM[name]]
        return lm.x * width, lm.y * height

    ls_x, ls_y = pt("left_shoulder")
    rs_x, rs_y = pt("right_shoulder")
    lh_x, lh_y = pt("left_hip")
    rh_x, rh_y = pt("right_hip")
    lk_x, lk_y = pt("left_knee")

    shoulder_y = (ls_y + rs_y) / 2
    hip_y = (lh_y + rh_y) / 2
    torso_h = hip_y - shoulder_y

    torso_x = (ls_x + rs_x + lh_x + rh_x) / 4
    thigh_x = lh_x + 0.25 * (lk_x - lh_x)
    thigh_y = lh_y + 0.25 * (lk_y - lh_y)

    return {
        "chest": (torso_x, int(shoulder_y + 0.25 * torso_h)),
        "waist": (torso_x, int(shoulder_y + 0.65 * torso_h)),
        "hip": (torso_x, int(hip_y)),
        "thigh": (thigh_x, int(thigh_y)),
    }


def ellipse_circumference(width_cm, depth_cm):
    """Ramanujan's approximation for the perimeter of an ellipse."""
    a = width_cm / 2.0
    b = depth_cm / 2.0
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def estimate_body_measurements(
    front_path,
    side_path,
    user_height_cm,
    model_path="pose_landmarker_heavy.task",
):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        output_segmentation_masks=True,
    )

    with vision.PoseLandmarker.create_from_options(options) as detector:
        f_lm, f_mask, f_h, f_w = analyze_image(front_path, detector)
        s_lm, s_mask, s_h, s_w = analyze_image(side_path, detector)

    # Calibrate each view separately in case camera distance changed slightly
    f_scale = get_cm_per_pixel(f_mask, user_height_cm)
    s_scale = get_cm_per_pixel(s_mask, user_height_cm)

    f_coords = get_measurement_coords(f_lm, f_h, f_w)
    s_coords = get_measurement_coords(s_lm, s_h, s_w)

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

    return measurements


if __name__ == "__main__":
    results = estimate_body_measurements(
        front_path="front.png",
        side_path="side.png",
        user_height_cm=175.0,
    )

    print("Estimated Body Measurements:")
    for part, data in results.items():
        print(
            f"  {part.capitalize():<6}: {data['circumference_cm']} cm "
            f"({data['circumference_in']} in) "
            f"[W: {data['width_cm']} cm, D: {data['depth_cm']} cm]"
        )