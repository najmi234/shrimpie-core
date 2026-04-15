import os
import random
import time
from collections import deque
from typing import Optional, Tuple, List

import cv2
import joblib
import pandas as pd
import numpy as np
from ultralytics import YOLO
from vidgear.gears import WriteGear
from centroid_tracker import CentroidTracker


# ============================================================
#  KONFIGURASI
# ============================================================

class Config:
    # Path model
    YOLO_MODEL_PATH       = "model/best.pt"
    LENGTH_MODEL_PATH     = "model/Newmodel_panjang.pkl"
    WEIGHT_MODEL_PATH     = "model/Newmodel_berat.pkl"

    # Threshold deteksi
    CONFIDENCE_THRESHOLD  = 0.80

    # Visualisasi mask
    MASK_ALPHA            = 0.8
    MASK_BLUE_CHANNEL     = 255
    MASK_GREEN_CHANNEL    = 150

    # Tracking
    MAX_TRAIL_LENGTH      = 50      # Batas titik jejak per objek

    # Warna
    COLOR_BBOX            = (255, 255, 0)   # Cyan
    COLOR_TEXT            = (0, 0, 0)

    # Kamera
    CAMERA_INDEX          = 0
    WINDOW_TITLE          = "Shrimp Detection"

    # Recording
    RECORD_DURATION = 10  # detik
    OUTPUT_DIR      = "recordings"


# ============================================================
#  LOADER MODEL
# ============================================================

def load_models(config: Config):
    """Memuat semua model yang dibutuhkan."""
    yolo_model     = YOLO(config.YOLO_MODEL_PATH)
    model_length   = joblib.load(config.LENGTH_MODEL_PATH)
    model_weight   = joblib.load(config.WEIGHT_MODEL_PATH)
    return yolo_model, model_length, model_weight


# ============================================================
#  UTILITAS WARNA
# ============================================================

def generate_random_color() -> tuple:
    """Menghasilkan warna BGR acak yang cukup terang."""
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )


# ============================================================
#  PEMROSESAN MASK & PREDIKSI
# ============================================================

def overlay_mask(frame: np.ndarray, mask_np: np.ndarray, config: Config) -> np.ndarray:
    """Menggabungkan warna mask ke frame."""
    colored_mask = np.zeros_like(frame)
    colored_mask[:, :, 0] = mask_np * config.MASK_BLUE_CHANNEL
    colored_mask[:, :, 1] = mask_np * config.MASK_GREEN_CHANNEL
    return cv2.addWeighted(frame, 1.0, colored_mask, config.MASK_ALPHA, 0)


def compute_measurements(
    contour: np.ndarray,
    mask_np: np.ndarray,
    model_length,
    model_weight,
) -> Tuple[float, float, np.ndarray]:
    """
    Menghitung panjang (cm) dan berat (gram) dari kontur udang.

    Returns:
        panjang_cm  : estimasi panjang dalam cm
        berat_gram  : estimasi berat dalam gram
        box_points  : 4 titik bounding box miring
    """
    area_px   = int(np.sum(mask_np))
    rect      = cv2.minAreaRect(contour)
    box       = np.intp(cv2.boxPoints(rect))

    edge_lengths = [
        np.linalg.norm(box[i] - box[(i + 1) % 4])
        for i in range(4)
    ]
    panjang_px = max(edge_lengths)

    panjang_cm = float(
        model_length.predict(pd.DataFrame([[panjang_px]], columns=["panjang_px"]))[0]
    )

    berat_gram = float(
        model_weight.predict(pd.DataFrame([[area_px]], columns=["area_px"]))[0]
    )

    return panjang_cm, berat_gram, box


def get_centroid(contour: np.ndarray) -> Optional[Tuple[int, int]]:
    """Menghitung centroid dari kontur. Mengembalikan None jika area = 0."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


# ============================================================
#  VISUALISASI
# ============================================================

def draw_bounding_box(
    frame: np.ndarray,
    box: np.ndarray,
    label: str,
    config: Config,
) -> None:
    """Menggambar bounding box miring dan label teks."""
    cv2.drawContours(frame, [box], 0, config.COLOR_BBOX, 1)
    x, y = box[0]
    cv2.putText(
        frame, label,
        (int(x), int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
        config.COLOR_TEXT, 1,
    )


def draw_tracking(
    frame: np.ndarray,
    object_id: int,
    centroid: tuple,
    color: tuple,
    trail: deque,
) -> None:
    """Menggambar ID, titik centroid, dan garis trajektori."""
    # Label ID
    cv2.putText(
        frame, f"ID {object_id}",
        (centroid[0] - 15, centroid[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1,
    )

    # Titik centroid
    cv2.circle(frame, centroid, 5, color, -1)

    # Garis trajektori
    pts = list(trail)
    for j in range(1, len(pts)):
        cv2.line(frame, pts[j - 1], pts[j], color, 2)

def letterbox_image(image, new_size=640):
    h, w = image.shape[:2]

    scale = min(new_size / w, new_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.ones((new_size, new_size, 3), dtype=np.uint8) * 255  # putih

    x_offset = (new_size - new_w) // 2
    y_offset = (new_size - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas, scale, x_offset, y_offset


# ============================================================
#  PIPELINE FRAME
# ============================================================

def process_frame(
    frame: np.ndarray,
    yolo_model,
    model_length,
    model_weight,
    tracker: CentroidTracker,
    track_history: dict,
    track_color: dict,
    config: Config,
) -> np.ndarray:

    # ========================
    # 1. LETTERBOX INPUT
    # ========================
    img_input, scale, x_offset, y_offset = letterbox_image(frame, 640)

    results = yolo_model(img_input, imgsz=640, verbose=False)
    result = results[0]

    frame_out = frame.copy()
    centroids = []

    masks = result.masks
    boxes = result.boxes

    if masks is not None:
        for i, mask in enumerate(masks.data):
            conf = float(boxes.conf[i])
            if conf < config.CONFIDENCE_THRESHOLD:
                continue

            # ========================
            # 2. MASK YOLO (640x640)
            # ========================
            mask_np = mask.cpu().numpy().astype(np.uint8)

            # ========================
            # 3. REMOVE PADDING
            # ========================
            h_scaled = int(frame.shape[0] * scale)
            w_scaled = int(frame.shape[1] * scale)

            mask_cropped = mask_np[
                y_offset:y_offset + h_scaled,
                x_offset:x_offset + w_scaled
            ]

            # ========================
            # 4. RESIZE KE FRAME ASLI
            # ========================
            mask_resized = cv2.resize(
                mask_cropped,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # ========================
            # 5. OVERLAY
            # ========================
            frame_out = overlay_mask(frame_out, mask_resized, config)

            # ========================
            # 6. CONTOUR (PAKAI YANG SUDAH RESIZED!)
            # ========================
            contours, _ = cv2.findContours(
                mask_resized,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)

            # ========================
            # 7. MEASUREMENT (VALID)
            # ========================
            panjang_cm, berat_gram, box = compute_measurements(
                contour, mask_resized, model_length, model_weight
            )

            label = f"{panjang_cm:.1f} cm | {berat_gram:.1f} g"
            draw_bounding_box(frame_out, box, label, config)

            # ========================
            # 8. CENTROID (VALID)
            # ========================
            centroid = get_centroid(contour)
            if centroid:
                centroids.append(centroid)

    # ========================
    # 9. TRACKING
    # ========================
    objects = tracker.update(centroids)

    for object_id, centroid in objects.items():
        if object_id not in track_color:
            track_color[object_id] = generate_random_color()

        if object_id not in track_history:
            track_history[object_id] = deque(maxlen=config.MAX_TRAIL_LENGTH)

        track_history[object_id].append(centroid)

        draw_tracking(
            frame_out,
            object_id,
            centroid,
            track_color[object_id],
            track_history[object_id],
        )

    frame_display = cv2.resize(frame_out, (640, 360))
    return frame_display


# ============================================================
#  LOOP UTAMA
# ============================================================

def run(config: Config = None) -> None:
    """Entry point utama aplikasi deteksi udang."""
    if config is None:
        config = Config()

    # Muat semua model
    yolo_model, model_length, model_weight = load_models(config)
    tracker       = CentroidTracker()
    track_history = {}
    track_color   = {}

    # Buka kamera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Kamera index {config.CAMERA_INDEX} tidak bisa dibuka!")

    print(f"🔥 Kamera aktif... tekan 'q' untuk keluar.")
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Resolusi kamera: {int(width)} x {int(height)}")

    # Recording state
    recording = False
    record_start_time = None
    video_writer = None

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Gagal membaca frame kamera!")
                break

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            frame_out = process_frame(
                frame,
                yolo_model, model_length, model_weight,
                tracker, track_history, track_color,
                config,
            )

            cv2.putText(
                frame_out,
                f"FPS: {fps:.1f}",
                (7, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                config.COLOR_TEXT,
                1,
            )

            cv2.imshow(config.WINDOW_TITLE, frame_out)

            # ========================
            # RECORDING LOGIC
            # ========================
            if recording:
                # VidGear menggunakan metode .write() yang sama dengan OpenCV
                video_writer.write(frame_out)

                elapsed = time.time() - record_start_time

                # Tampilkan indikator recording
                cv2.putText(
                    frame_out,
                    f"REC {elapsed:.1f}s",
                    (500, 30), # Sedikit turun agar tidak terpotong
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

                if elapsed >= config.RECORD_DURATION:
                    print("✅ Recording selesai!")
                    recording = False
                    # VidGear menggunakan .close() bukan .release()
                    video_writer.close()
                    video_writer = None

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p") and not recording:
                print("⏺️ Recording started (H.264 Web-Ready)...")

                recording = True
                record_start_time = time.time()

                filename = time.strftime("%d%m%Y-%H%M") + ".mp4"
                filepath = os.path.join(config.OUTPUT_DIR, filename)

                # Definisikan parameter FFmpeg agar video bisa diputar di Website
                output_params = {
                    "-vcodec": "libx264",    # Codec standar browser
                    "-crf": 20,              # Kualitas (18-28 adalah sweet spot)
                    "-preset": "veryfast",   # Ringan untuk CPU
                    "-pix_fmt": "yuv420p",   # WAJIB untuk kompatibilitas web
                    "-movflags": "+faststart" # Metadata di depan agar video cepat loading
                }

                # Inisialisasi WriteGear (Ganti cv2.VideoWriter)
                video_writer = WriteGear(output=filepath, **output_params)

            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Kamera dilepas, program selesai.")


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()
