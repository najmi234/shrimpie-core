import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from centroid_tracker import CentroidTracker
import random
from scipy.spatial import distance as dist

# ============================
#  Load model regresi
# ============================
model_len = joblib.load("model/Newmodel_panjang.pkl")
model_weight = joblib.load("model/Newmodel_berat.pkl")

# ============================
#  Load YOLOv11 segmentation
# ============================
yolo_model = YOLO("model/best.pt")
tracker = CentroidTracker()

# ============================
#  Tempat menyimpan jejak ID
# ============================
track_history = {}       # {id: [(x1,y1), (x2,y2), ...]}
track_color = {}         # {id: (B,G,R)}

# ============================
#  Data aktivitas udang
# ============================
shrimp_activity = {}     # {id: {"prev":(x,y), "distance":float, "speed":float}}
FPS = 30                 # Sesuaikan dengan kamera kamu

def get_random_color():
    return (random.randint(50,255), random.randint(50,255), random.randint(50,255))

# ============================
#  Buka kamera
# ============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera tidak bisa dibuka!")
    exit()

print("🔥 Kamera aktif... tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame kamera!")
        break

    # Predict YOLO
    results = yolo_model(frame)
    result = results[0]

    frame_output = frame.copy()

    masks = result.masks
    boxes = result.boxes

    centroids = []

    # ============================
    #  YOLO SEGMENTATION
    # ============================
    if masks is not None:
        for i, mask in enumerate(masks.data):
            conf = float(boxes.conf[i])
            if conf < 0.80:
                continue

            mask_np = mask.cpu().numpy().astype(np.uint8)

            # warna masker
            colored_mask = np.zeros_like(frame_output)
            colored_mask[:, :, 0] = mask_np * 255
            colored_mask[:, :, 1] = mask_np * 150
            frame_output = cv2.addWeighted(frame_output, 1.0, colored_mask, 0.8, 0)

            # AREA
            area_px = np.sum(mask_np)

            # MIN AREA RECTANGLE
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = contours[0]
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                cv2.drawContours(frame_output, [box], 0, (255, 255, 0), 2)

                # Panjang px
                edge_lengths = [np.linalg.norm(box[i] - box[(i+1) % 4]) for i in range(4)]
                panjang_px = max(edge_lengths)

                # Prediksi panjang & berat
                panjang_cm = model_len.predict([[panjang_px]])[0]
                berat_gram = model_weight.predict([[area_px]])[0]

                x, y = box[0]
                text = f"{panjang_cm:.1f} cm | {berat_gram:.1f} g"
                cv2.putText(frame_output, text, (int(x), int(y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                # Centroid
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroids.append((cX, cY))

    # ============================
    #  UPDATE TRACKER
    # ============================
    objects = tracker.update(centroids)

    # ============================
    #  DRAWING + ACTIVITY ANALYSIS
    # ============================
    for objectID, centroid in objects.items():

        # === WARNA UNTUK ID ===
        if objectID not in track_color:
            track_color[objectID] = get_random_color()

        # === TRAJECTORY ===
        if objectID not in track_history:
            track_history[objectID] = []
        track_history[objectID].append(centroid)

        # === INISIALISASI AKTIVITAS ===
        if objectID not in shrimp_activity:
            shrimp_activity[objectID] = {
                "prev": centroid,
                "distance": 0.0,
                "speed": 0.0,
            }

        # === HITUNG JARAK ===
        prev_x, prev_y = shrimp_activity[objectID]["prev"]
        cur_x, cur_y = centroid

        dist_move = np.sqrt((cur_x - prev_x)**2 + (cur_y - prev_y)**2)
        shrimp_activity[objectID]["distance"] += dist_move

        # kecepatan px/detik
        shrimp_activity[objectID]["speed"] = dist_move * FPS

        # update posisi sebelumnya
        shrimp_activity[objectID]["prev"] = centroid

        # SHOW ID
        cv2.putText(frame_output, f"ID {objectID}",
                    (centroid[0] - 15, centroid[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    track_color[objectID], 2)

        # titik centroid
        cv2.circle(frame_output, (centroid[0], centroid[1]), 5,
                   track_color[objectID], -1)

        # garis trajectory
        pts = track_history[objectID]
        if len(pts) > 2:
            for j in range(1, len(pts)):
                cv2.line(frame_output, pts[j - 1], pts[j], track_color[objectID], 2)

        # ============================
        #  TAMPILKAN INFORMASI AKTIVITAS
        # ============================
        act_speed = shrimp_activity[objectID]["speed"]

        cv2.putText(frame_output,
                    f"Act: {act_speed:.1f}px/s",
                    (centroid[0] + 20, centroid[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    track_color[objectID],
                    2)

    # ============================
    #  TAMPILKAN HASIL
    # ============================
    cv2.imshow("Shrimp Detection", frame_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
