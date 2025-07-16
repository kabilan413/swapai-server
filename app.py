from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os
import uuid
import cv2
from werkzeug.utils import secure_filename
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = Flask(__name__)
CORS(app)

# Setup folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("temp_frames", exist_ok=True)

# Load InsightFace models
print("🔍 Loading InsightFace models...")
faceapp = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0)
swapper = get_model("inswapper_128.onnx", download=True, providers=['CPUExecutionProvider'])
print("✅ Models loaded.")

@app.route('/')
def index():
    return jsonify({"message": "SwapAI server is running on Render!"})

@app.route('/templates')
def get_templates():
    try:
        templates = []
        for filename in os.listdir("templates"):
            if filename.endswith(".mp4"):
                template_id = filename.replace("template", "").replace(".mp4", "")
                thumb = f"thumb_template{template_id}.jpg"
                templates.append({
                    "id": int(template_id),
                    "name": f"Template {template_id}",
                    "video": f"/templates/{filename}",
                    "thumb": f"/templates/{thumb}"
                })
        templates.sort(key=lambda x: x["id"])
        return jsonify(templates)
    except Exception as e:
        print(f"❌ Failed to list templates: {e}")
        return jsonify([])

@app.route('/swap', methods=['POST'])
def swap_face():
    try:
        uploaded_file = request.files['file']
        template_name = request.form.get("template")
        if not template_name:
            return "Template name is required", 400

        template_path = os.path.join("templates", template_name)
        if not os.path.exists(template_path):
            return "Template video not found", 404

        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join("uploads", filename)
        uploaded_file.save(filepath)

        img_src = cv2.imread(filepath)
        faces_src = faceapp.get(img_src)
        if len(faces_src) == 0:
            return "No face detected", 400
        source_face = faces_src[0]

        cap = cv2.VideoCapture(template_path)
        temp_dir = os.path.join("temp_frames", str(uuid.uuid4()))
        os.makedirs(temp_dir, exist_ok=True)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces_dst = faceapp.get(frame)
            for target_face in faces_dst:
                frame = swapper.get(frame, target_face, source_face, paste_back=True)
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        cap.release()

        output_path = os.path.join("static", f"output_{uuid.uuid4().hex}.mp4")
        ffmpeg_cmd = f"ffmpeg -y -framerate 25 -i {temp_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p {output_path}"
        os.system(ffmpeg_cmd)

        return send_file(output_path, mimetype='video/mp4')
    except Exception as e:
        print(f"❌ Error: {e}")
        return "Internal server error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
