import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# BlazePose 33-point landmark indices
LM = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


# =========================================
# 1. POSE & SILHOUETTE EXTRACTION
# =========================================

def analyze_image(image_path, detector):
    """Run PoseLandmarker and return the resized BGR image, landmarks, and binary mask."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

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

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return image, landmarks, mask, new_h, new_w


def get_cm_per_pixel(mask, user_height_cm):
    ys = np.where(mask > 0)[0]
    if len(ys) == 0:
        raise ValueError("Empty segmentation mask")
    return user_height_cm / int(ys.max() - ys.min())


def measure_segment_width_px(mask, y_center, x_ref, band=3):
    h, w = mask.shape
    widths = []

    for y in range(max(0, y_center - band), min(h, y_center + band + 1)):
        row = mask[y, :]
        x = int(np.clip(x_ref, 0, w - 1))

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
        "shoulder": (torso_x, int(shoulder_y)),
        "chest": (torso_x, int(shoulder_y + 0.25 * torso_h)),
        "waist": (torso_x, int(shoulder_y + 0.65 * torso_h)),
        "hip": (torso_x, int(hip_y)),
        "thigh": (thigh_x, int(thigh_y)),
    }


def ellipse_circumference(width_cm, depth_cm):
    a = width_cm / 2.0
    b = depth_cm / 2.0
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


# =========================================
# 2. POSE VALIDATION & WEIGHT CALIBRATION
# =========================================

def validate_pose_and_angle(landmarks, view="front"):
    """
    Validate that the photo shows the full body and matches the requested angle.
    Returns (is_valid: bool, warnings: list[str]).
    """
    warnings = []

    nose = landmarks[LM["nose"]]
    l_ankle, r_ankle = landmarks[LM["left_ankle"]], landmarks[LM["right_ankle"]]
    l_sh, r_sh = landmarks[LM["left_shoulder"]], landmarks[LM["right_shoulder"]]
    l_hip, r_hip = landmarks[LM["left_hip"]], landmarks[LM["right_hip"]]

    if nose.visibility < 0.5:
        warnings.append("Head/face is not clearly visible. Step back slightly.")
    if max(l_ankle.visibility, r_ankle.visibility) < 0.5:
        warnings.append("Feet/ankles are cut off. Step back ~2 meters so your full body fits.")

    shoulder_dx = abs(l_sh.x - r_sh.x)
    torso_dy = max(abs(((l_hip.y + r_hip.y) / 2) - ((l_sh.y + r_sh.y) / 2)), 0.01)
    shoulder_ratio = shoulder_dx / torso_dy

    if view == "front" and shoulder_ratio < 0.35:
        warnings.append("You appear turned sideways. Face straight toward the camera for the Front photo.")
    elif view == "side" and shoulder_ratio > 0.32:
        warnings.append("You are not fully sideways. Turn a full 90 degrees for the Side photo.")

    return (len(warnings) == 0), warnings


def apply_weight_calibration(measurements, height_cm, weight_kg, gender="male"):
    """
    Use height + weight to detect baggy clothes or camera distortion,
    and blend silhouette measurements with anthropometric expectations.
    """
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)

    if gender.lower() == "female":
        expected_waist = 0.38 * height_cm + 1.65 * (bmi - 22.0)
        expected_chest = 0.50 * height_cm + 1.50 * (bmi - 22.0)
        expected_hip = 0.53 * height_cm + 1.60 * (bmi - 22.0)
    else:
        expected_waist = 0.45 * height_cm + 1.95 * (bmi - 22.0)
        expected_chest = 0.55 * height_cm + 1.65 * (bmi - 22.0)
        expected_hip = 0.54 * height_cm + 1.45 * (bmi - 22.0)

    priors = {
        "chest": expected_chest,
        "waist": expected_waist,
        "hip": expected_hip,
    }

    raw_waist = measurements["waist"]["circumference_cm"]
    waist_diff = raw_waist - expected_waist

    alerts = []
    alpha = 0.80

    if waist_diff > 10.0:
        alerts.append(
            f"Baggy clothing detected: silhouette waist ({raw_waist} cm) is much larger than "
            f"expected for {weight_kg} kg. Measurements were adjusted using your weight."
        )
        alpha = 0.60
    elif waist_diff < -10.0:
        alerts.append(
            "Possible camera tilt or distance distortion: silhouette looks smaller than expected "
            "for your height & weight. Retake with the camera straight at waist height."
        )
        alpha = 0.65

    calibrated = {}
    for part, data in measurements.items():
        raw_c = data["circumference_cm"]
        if part in priors:
            adj_c = alpha * raw_c + (1.0 - alpha) * priors[part]
        else:
            adj_c = raw_c

        calibrated[part] = {
            **data,
            "raw_circumference_cm": raw_c,
            "circumference_cm": round(adj_c, 1),
            "circumference_in": round(adj_c / 2.54, 1),
        }

    return calibrated, alerts


# =========================================
# 3. CIELAB SKIN-TONE & COLOR ANALYSIS
# =========================================

def gray_world_white_balance(bgr_image):
    """Neutralize warm/cool indoor lighting cast before skin sampling."""
    result = bgr_image.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(result, axis=(0, 1))
    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    result[:, :, 0] *= avg_gray / (avg_b + 1e-6)
    result[:, :, 1] *= avg_gray / (avg_g + 1e-6)
    result[:, :, 2] *= avg_gray / (avg_r + 1e-6)

    return np.clip(result, 0, 255).astype(np.uint8)


def analyze_skin_tone(bgr_image, landmarks):
    """Sample cheek & forehead patches, convert to CIELAB, and return color palette."""
    balanced = gray_world_white_balance(bgr_image)
    h, w = balanced.shape[:2]

    def pt(name):
        lm = landmarks[LM[name]]
        return int(lm.x * w), int(lm.y * h)

    nose = pt("nose")
    l_eye = pt("left_eye")
    r_eye = pt("right_eye")
    l_ear = pt("left_ear")
    r_ear = pt("right_ear")

    eye_dist = max(10, int(math.hypot(l_eye[0] - r_eye[0], l_eye[1] - r_eye[1])))
    patch_r = max(4, int(eye_dist * 0.22))

    eye_mid_x = (l_eye[0] + r_eye[0]) // 2
    eye_mid_y = (l_eye[1] + r_eye[1]) // 2

    sample_centers = [
        ((nose[0] + l_ear[0]) // 2, (nose[1] + l_eye[1]) // 2),
        ((nose[0] + r_ear[0]) // 2, (nose[1] + r_eye[1]) // 2),
        (eye_mid_x, max(patch_r, eye_mid_y - int(eye_dist * 0.7))),
    ]

    lab_pixels = []
    lab_img = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)

    for cx, cy in sample_centers:
        y1, y2 = max(0, cy - patch_r), min(h, cy + patch_r)
        x1, x2 = max(0, cx - patch_r), min(w, cx + patch_r)
        patch = lab_img[y1:y2, x1:x2]
        if patch.size > 0:
            lab_pixels.append(patch.reshape(-1, 3))

    if not lab_pixels:
        return {"error": "Could not sample facial skin patches"}

    all_pixels = np.vstack(lab_pixels).astype(np.float32)

    L_star = float(np.median(all_pixels[:, 0]) * (100.0 / 255.0))
    a_star = float(np.median(all_pixels[:, 1]) - 128.0)
    b_star = float(np.median(all_pixels[:, 2]) - 128.0)

    if L_star >= 70:
        depth = "Fair / Light"
    elif L_star >= 55:
        depth = "Medium"
    elif L_star >= 40:
        depth = "Tan / Olive"
    else:
        depth = "Deep / Rich"

    if b_star >= 15.0 and (b_star - a_star) >= 4.0:
        undertone = "Warm"
    elif b_star <= 10.0 or a_star > b_star:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    if undertone == "Warm" and L_star >= 55:
        season = "Spring (Warm & Light)"
        best_colors = ["Coral", "Warm Peach", "Turquoise", "Cream", "Camel", "Mint Green"]
        avoid_colors = ["Stark Black", "Icy Blue", "Dark Charcoal"]
    elif undertone == "Warm":
        season = "Autumn (Warm & Deep)"
        best_colors = ["Olive Green", "Mustard", "Rust", "Terracotta", "Warm Brown", "Teal"]
        avoid_colors = ["Pastel Pink", "Neon Blue", "Pure White"]
    elif undertone == "Cool" and L_star >= 55:
        season = "Summer (Cool & Light)"
        best_colors = ["Lavender", "Powder Blue", "Dusty Rose", "Soft Navy", "Cool Gray"]
        avoid_colors = ["Orange", "Mustard Yellow", "Warm Brown"]
    elif undertone == "Cool":
        season = "Winter (Cool & Deep)"
        best_colors = ["Emerald Green", "Royal Blue", "Burgundy", "Pure White", "True Black"]
        avoid_colors = ["Beige", "Orange", "Earthy Olive"]
    else:
        season = "Neutral (Balanced)"
        best_colors = ["Jade Green", "Dusty Pink", "Soft White", "Navy", "Teal", "Taupe"]
        avoid_colors = ["Overly Neon Yellow", "Harsh Fluorescent Orange"]

    return {
        "cielab": {"L": round(L_star, 1), "a": round(a_star, 1), "b": round(b_star, 1)},
        "depth": depth,
        "undertone": undertone,
        "season": season,
        "best_colors": best_colors,
        "avoid_colors": avoid_colors,
    }


# =========================================
# 4. CLOTHING SIZE & BODY SHAPE ENGINE
# =========================================

def map_clothing_size(measurements, gender="male"):
    """Map cm circumferences to standard Top & Bottom sizes with fit notes."""
    chest = measurements["chest"]["circumference_cm"]
    waist = measurements["waist"]["circumference_cm"]
    hip = measurements["hip"]["circumference_cm"]
    thigh = measurements["thigh"]["circumference_cm"]

    if gender.lower() == "female":
        top_chart = [(82, "XS"), (88, "S"), (94, "M"), (102, "L"), (110, "XL")]
        bottom_chart = [(88, "XS"), (94, "S"), (100, "M"), (108, "L"), (116, "XL")]
        bottom_metric = hip
    else:
        top_chart = [(88, "XS"), (96, "S"), (104, "M"), (112, "L"), (120, "XL")]
        bottom_chart = [(74, "XS"), (81, "S"), (89, "M"), (97, "L"), (105, "XL")]
        bottom_metric = waist

    def lookup(val, chart):
        for limit, label in chart:
            if val <= limit:
                return label
        return "XXL"

    top_size = lookup(chest, top_chart)
    bottom_size = lookup(bottom_metric, bottom_chart)
    waist_in = round(waist / 2.54)

    fit_notes = []
    if thigh / max(waist, 1) > 0.68:
        fit_notes.append("Athletic/Relaxed-thigh trousers recommended (fuller thigh-to-waist ratio).")
    if waist / max(chest, 1) > 0.92:
        fit_notes.append("Consider sizing up on slim-fit shirts or choosing regular/relaxed-fit tops.")

    return {
        "top_size": top_size,
        "bottom_size": f"{bottom_size} (Waist ~{waist_in} in)",
        "fit_notes": fit_notes or ["Standard regular fit should align well with your proportions."],
    }


def classify_body_shape(measurements, shoulder_width_cm):
    """Classify body shape and return tailored clothing style recommendations."""
    chest = measurements["chest"]["circumference_cm"]
    waist = measurements["waist"]["circumference_cm"]
    hip = measurements["hip"]["circumference_cm"]
    hip_width = measurements["hip"]["width_cm"]

    shoulder_to_hip = shoulder_width_cm / max(hip_width, 1)
    waist_to_hip = waist / max(hip, 1)
    waist_to_chest = waist / max(chest, 1)

    if waist_to_hip <= 0.78 and waist_to_chest <= 0.80:
        shape = "Hourglass"
        styles = [
            "Fitted or tailored silhouettes that highlight the waist",
            "Wrap tops, belted jackets, and high-waisted trousers",
            "Avoid overly boxy or shapeless oversized cuts",
        ]
    elif shoulder_to_hip < 0.95 or (hip - chest) >= 8:
        shape = "Pear / Triangle"
        styles = [
            "Structured shoulders, layered jackets, and boat/V-neck tops to balance proportions",
            "Darker, straight-leg or bootcut trousers",
            "Avoid tight tapered pants that cling to the hips and thighs",
        ]
    elif shoulder_to_hip > 1.15 or (chest - hip) >= 10:
        shape = "Inverted Triangle"
        styles = [
            "V-neck tees and clean, minimal upper-body layers (avoid heavy shoulder pads)",
            "Straight-fit, relaxed, or pleated trousers to add balance to the lower half",
            "Lighter colors or subtle textures on bottoms",
        ]
    elif waist_to_chest >= 0.93 or waist_to_hip >= 0.93:
        shape = "Oval / Apple"
        styles = [
            "Structured open layers (overshirts, blazers) to create clean vertical lines",
            "Mid-rise straight-leg trousers and breathable regular-fit tops",
            "Avoid clingy fabrics or tight belts across the midsection",
        ]
    else:
        shape = "Rectangle / Athletic"
        styles = [
            "Layered outfits, textured fabrics, and structured jackets to add dimension",
            "Slim-straight or tapered trousers",
            "Color-blocking between top and bottom works very well",
        ]

    return {"body_shape": shape, "recommended_styles": styles}


# =========================================
# 5. STANDALONE CLI PIPELINE
# =========================================

def run_full_analysis(
    front_path,
    side_path,
    user_height_cm,
    user_weight_kg,
    gender="male",
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
        f_img, f_lm, f_mask, f_h, f_w = analyze_image(front_path, detector)
        _, s_lm, s_mask, s_h, s_w = analyze_image(side_path, detector)

    f_scale = get_cm_per_pixel(f_mask, user_height_cm)
    s_scale = get_cm_per_pixel(s_mask, user_height_cm)

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

    measurements, alerts = apply_weight_calibration(
        raw_measurements, user_height_cm, user_weight_kg, gender=gender
    )

    bmi = round(user_weight_kg / ((user_height_cm / 100.0) ** 2), 1)
    color_profile = analyze_skin_tone(f_img, f_lm)
    size_profile = map_clothing_size(measurements, gender=gender)
    style_profile = classify_body_shape(measurements, shoulder_width_cm)

    return {
        "bmi": bmi,
        "alerts": alerts,
        "measurements": measurements,
        "sizing": size_profile,
        "style": style_profile,
        "color_analysis": color_profile,
    }


if __name__ == "__main__":
    report = run_full_analysis(
        front_path="front.jpg",
        side_path="side.jpg",
        user_height_cm=175.0,
        user_weight_kg=72.0,
        gender="male",
    )

    if report["alerts"]:
        print("\n=== CALIBRATION ALERTS ===")
        for alert in report["alerts"]:
            print(f"  ! {alert}")

    print("\n=== BODY MEASUREMENTS ===")
    for part, d in report["measurements"].items():
        print(f"  {part.capitalize():<6}: {d['circumference_cm']} cm ({d['circumference_in']} in)")

    print("\n=== CLOTHING SIZE ===")
    print(f"  Top Size   : {report['sizing']['top_size']}")
    print(f"  Bottom Size: {report['sizing']['bottom_size']}")

    print("\n=== BODY SHAPE & STYLE ===")
    print(f"  Body Shape : {report['style']['body_shape']}")
    for tip in report["style"]["recommended_styles"]:
        print(f"  - {tip}")

    print("\n=== SKIN TONE & COLOR PALETTE ===")
    c = report["color_analysis"]
    print(f"  Season     : {c['season']}")
    print(f"  Best Colors: {', '.join(c['best_colors'])}")