import cv2
import numpy as np
import joblib
from ultralytics import YOLO

# ============================
#  Load model regresi
# ============================
model_len = joblib.load("model/Newmodel_panjang.pkl")
model_weight = joblib.load("model/Newmodel_berat.pkl")

# ============================
#  Load YOLOv11 segmentation
# ============================
yolo_model = YOLO("model/best.pt")

# ============================
#  Buka kamera
# ============================
cap = cv2.VideoCapture(2)

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

    # Frame yang akan digambar
    frame_output = frame.copy()

    masks = result.masks
    boxes = result.boxes

    if masks is not None:
        for i, mask in enumerate(masks.data):
            conf = float(boxes.conf[i])
            if conf < 0.50:
                continue

            # ---------------------------
            #   GAMBAR MASK MANUAL
            # ---------------------------
            mask_np = mask.cpu().numpy().astype(np.uint8)
            colored_mask = np.zeros_like(frame_output)
            colored_mask[:, :, 0] = mask_np * 255  # BLUE
            colored_mask[:, :, 1] = mask_np * 150  # GREEN
            frame_output = cv2.addWeighted(frame_output, 1.0, colored_mask, 0.8, 0)

            # ---------------------------
            #   HITUNG KONTUR & BOX
            # ---------------------------
            area_px = np.sum(mask_np)
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = contours[0]
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                cv2.drawContours(frame_output, [box], 0, (255, 255, 0), 2)

                edge_lengths = [np.linalg.norm(box[i] - box[(i+1) % 4]) for i in range(4)]
                panjang_px = max(edge_lengths)

                # Prediksi
                panjang_cm = model_len.predict([[panjang_px]])[0]
                berat_gram = model_weight.predict([[area_px]])[0]

                text = f"{panjang_cm:.1f} cm | {berat_gram:.1f} g"
                x, y = box[0]
                cv2.putText(frame_output, text, (int(x), int(y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("YOLO Shrimp Detection + Prediction", frame_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
