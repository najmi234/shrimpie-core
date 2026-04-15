from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Gunakan path absolut agar lebih aman di production
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "recordings")

# Pastikan folder ada agar tidak error saat start
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

@app.route("/")
def home():
    return {"message": "Shrimp Video Server Running"}

@app.route("/videos")
def list_videos():
    try:
        video_list = []
        files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
        
        for filename in files:
            name_part = filename.replace(".mp4", "")
            parts = name_part.split("_")

            raw_timestamp = ""
            raw_device_id = ""

            for part in parts:
                if "web" in part:
                    raw_device_id = part
                elif "-" in part:
                    raw_timestamp = part

            display_device_name = raw_device_id.replace("web", "jetson ")
            display_device_id = "1f42a0df-72fb-4c52-88b3-58b92d824fe3"

            try:
                dt = datetime.strptime(raw_timestamp, "%d%m%Y-%H%M")
                recorded_at = dt.isoformat()
            except:
                recorded_at = datetime.now().isoformat()

            video_list.append({
                "device_id": display_device_id,
                "device_name": display_device_name,
                "file_url": f"https://shrimpie.qzz.io/video/{filename}",
                "recorded_at": recorded_at,
            })

        video_list.sort(key=lambda x: x['recorded_at'], reverse=True)
        return jsonify(video_list)

    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/video/<filename>")
def get_video(filename):
    try:
        # send_from_directory sudah cukup aman untuk streaming dasar
        return send_from_directory(VIDEO_DIR, filename)
    except Exception as e:
        return {"error": str(e)}, 404

# Blok ini akan diabaikan oleh Gunicorn, tapi tetap berguna jika ingin debug lokal
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
