import cv2
import time

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Kamera tidak ditemukan atau gagal dibuka.")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30

recording = False
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if recording and out is not None:
        out.write(frame)

    cv2.imshow("USB Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        if not recording:
            filename = f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            recording = True
            print(f"Mulai merekam video MP4: {filename}")
        else:
            recording = False
            out.release()
            out = None
            print("Rekaman dihentikan.")

    elif key == 27:
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
