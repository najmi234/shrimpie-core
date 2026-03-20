import cv2
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak ditemukan atau gagal dibuka.")
    exit()


# Loop utama
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame.")
        break

    # Tampilkan video
    cv2.imshow("USB Camera", frame)

    # Baca input keyboard
    key = cv2.waitKey(1) & 0xFF

    # Tekan 's' untuk menyimpan gambar
    if key == ord('s'):
        filename = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Gambar disimpan sebagai: {filename}")
    
    # Tekan ESC untuk keluar
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
