from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

VIDEO_DIR = "recordings"


@app.route("/")
def home():
    return {"message": "Shrimp Video Server Running"}


# =========================
# LIST VIDEO
# =========================
@app.route("/videos")
def list_videos():
    try:
        files = [
            f for f in os.listdir(VIDEO_DIR)
            if f.endswith(".mp4")
        ]
        files.sort(reverse=True)
        return jsonify(files)
    except Exception as e:
        return {"error": str(e)}, 500


# =========================
# STREAM VIDEO
# =========================
@app.route("/video/<filename>")
def get_video(filename):
    try:
        return send_from_directory(VIDEO_DIR, filename)
    except Exception as e:
        return {"error": str(e)}, 404


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)