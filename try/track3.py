import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from centroid_tracker import CentroidTracker
import random
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import pandas as pd
import time

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
#  Tracking data
# ============================
track_history = {}
track_color = {}

# ============================
#  Aktivitas udang
# ============================
shrimp_activity = {}   
activity_graph = {}    # per ID: list kecepatan untuk grafik
FPS = 30
ACT_THRESHOLD = 20     # px/s → threshold aktif

start_time = time.time()

# CSV log
csv_file = "aktivitas_log.csv"
df_log = pd.DataFrame(columns=["timestamp", "ID", "distance", "avg_speed", "status"])
df_log.to_csv(csv_file, index=False)


def get_random_color():
    return (random.randint(50,255), random.randint(50,255), random.randint(50,255))


# ============================
#  Grafik Real-Time
# ============================
plt.ion()
fig, ax = plt.subplots()
ax.set_title("Grafik Aktivitas Real-Time (Kecepatan px/s)")
ax.set_xlabel("Frame ke-")
ax.set_ylabel("Kecepatan px/s")

lines = {}  # line plot per ID


def update_graph():
    for obj_id, speeds in activity_graph.items():
        if obj_id not in lines:
            (line,) = ax.plot(speeds, label=f"ID {obj_id}")
            lines[obj_id] = line
            ax.legend()
        else:
            lines[obj_id].set_ydata(speeds)
            lines[obj_id].set_xdata(range(len(speeds)))

    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)


# ============================
#  Kamera
# ============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera tidak bisa dibuka!")
    exit()

print("🔥 Kamera aktif... tekan 'q' untuk keluar.")


# ============================
#  MAIN LOOP
# ============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Gagal membaca frame kamera!")
        break

    results = yolo_model(frame)
    result = results[0]

    frame_output = frame.copy()

    masks = result.masks
    boxes = result.boxes
    centroids = []

    # ============================
    #  YOLO segmentation
    # ============================
    if masks is not None:
        for i, mask in enumerate(masks.data):
            conf = float(boxes.conf[i])
            if conf < 0.90:
                continue

            mask_np = mask.cpu().numpy().astype(np.uint8)

            colored_mask = np.zeros_like(frame_output)
            colored_mask[:, :, 0] = mask_np * 255
            colored_mask[:, :, 1] = mask_np * 150
            frame_output = cv2.addWeighted(frame_output, 1, colored_mask, 0.8, 0)

            area_px = np.sum(mask_np)

            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = contours[0]
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(frame_output, [box], 0, (255,255,0), 2)

                edge_lengths = [np.linalg.norm(box[i] - box[(i+1)%4]) for i in range(4)]
                panjang_px = max(edge_lengths)

                panjang_cm = model_len.predict([[panjang_px]])[0]
                berat_gram = model_weight.predict([[area_px]])[0]

                x, y = box[0]
                cv2.putText(frame_output, f"{panjang_cm:.1f} cm | {berat_gram:.1f} g",
                            (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255,255,0), 2)

                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    centroids.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))


    # ============================
    #  Update tracker
    # ============================
    objects = tracker.update(centroids)

    # ============================
    #  Tracking + Aktivitas
    # ============================
    for objectID, centroid in objects.items():

        if objectID not in track_color:
            track_color[objectID] = get_random_color()

        if objectID not in track_history:
            track_history[objectID] = []
        track_history[objectID].append(centroid)

        if objectID not in shrimp_activity:
            shrimp_activity[objectID] = {
                "prev": centroid,
                "distance": 0.0,
                "speed": 0.0,
            }

        if objectID not in activity_graph:
            activity_graph[objectID] = []

        # hitung jarak berpindah
        px, py = shrimp_activity[objectID]["prev"]
        cx, cy = centroid
        dist_move = np.sqrt((cx - px)**2 + (cy - py)**2)

        # simpan
        shrimp_activity[objectID]["distance"] += dist_move
        shrimp_activity[objectID]["speed"] = dist_move * FPS
        shrimp_activity[objectID]["prev"] = centroid

        # simpan untuk grafik
        activity_graph[objectID].append(shrimp_activity[objectID]["speed"])

        # klasifikasi
        status = "AKTIF" if shrimp_activity[objectID]["speed"] > ACT_THRESHOLD else "TIDAK AKTIF"

        # tampilkan data aktivitas
        cv2.putText(frame_output,
                    f"ID {objectID} | {status} | {shrimp_activity[objectID]['speed']:.1f}px/s",
                    (centroid[0]-10, centroid[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    track_color[objectID], 2)

        cv2.circle(frame_output, centroid, 5, track_color[objectID], -1)

        pts = track_history[objectID]
        if len(pts) > 2:
            for j in range(1, len(pts)):
                cv2.line(frame_output, pts[j-1], pts[j], track_color[objectID], 2)

    # ============================
    #  Update Grafik
    # ============================
    update_graph()

    # ============================================================
    #  ANALISIS KESELURUHAN AKTIVITAS UDANG
    # ============================================================
    total_udang = len(shrimp_activity)
    aktif_count = 0

    for obj_id, act in shrimp_activity.items():
        if act["speed"] > ACT_THRESHOLD:
            aktif_count += 1

    tidak_aktif_count = total_udang - aktif_count

    # Hindari error bila belum terdeteksi
    if total_udang == 0:
        persentase_aktif = 0
    else:
        persentase_aktif = (aktif_count / total_udang) * 100

    # Klasifikasi aktivitas global
    if persentase_aktif > 60:
        status_global = "AKTIVITAS TINGGI"
        warna_global = (0, 255, 255)  # kuning
    elif persentase_aktif > 20:
        status_global = "AKTIVITAS NORMAL"
        warna_global = (0, 255, 0)    # hijau
    else:
        status_global = "AKTIVITAS RENDAH"
        warna_global = (0, 0, 255)    # merah

    # Tampilkan di layar
    cv2.rectangle(frame_output, (10, 10), (370, 120), (0, 0, 0), -1)
    cv2.putText(frame_output, f"Aktif: {aktif_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame_output, f"Tidak Aktif: {tidak_aktif_count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame_output, f"Status: {status_global}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna_global, 2)

    # ============================
    #  Logging ke CSV tiap 60 detik
    # ============================
    if time.time() - start_time >= 60:
        print("📝 Menyimpan aktivitas ke CSV...")
        rows = []

        for obj_id, act in shrimp_activity.items():
            rows.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ID": obj_id,
                "distance": act["distance"],
                "avg_speed": np.mean(activity_graph[obj_id][-FPS*60:]) if len(activity_graph[obj_id]) > FPS*60 else 0,
                "status": "AKTIF" if act["speed"] > ACT_THRESHOLD else "TIDAK AKTIF"
            })

        pd.DataFrame(rows).to_csv(csv_file, mode='a', header=False, index=False)
        start_time = time.time()

    # ============================
    #  Tampilkan
    # ============================
    cv2.imshow("Shrimp Detection", frame_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
plt.close()
