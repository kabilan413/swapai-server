from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
TEMPLATE_FOLDER = "templates"
STATIC_FOLDER = "static"
TEMP_FRAMES_FOLDER = "temp_frames"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(TEMP_FRAMES_FOLDER, exist_ok=True)

faceapp = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0)
swapper = get_model("inswapper_128.onnx", download=True, providers=['CPUExecutionProvider'])

@app.route('/')
def index():
    return jsonify({"message": "SwapAI AI Server v2 is running!"})

@app.route('/templates')
def get_templates():
    templates = []
    for i in range(1, 10):
        video_file = f"{TEMPLATE_FOLDER}/template{i}.mp4"
        thumb_file = f"{TEMPLATE_FOLDER}/thumb_template{i}.jpg"
        if os.path.exists(video_file) and os.path.exists(thumb_file):
            templates.append({
                "id": i,
                "video": f"/{video_file}",
                "thumb": f"/{thumb_file}",
                "name": f"Template {i}"
            })
    return jsonify(templates)

@app.route('/swap', methods=['POST'])
def swap_face():
    try:
        uploaded_file = request.files['file']
        template_name = request.form.get("template", "template1.mp4")
        template_path = os.path.join(TEMPLATE_FOLDER, template_name)

        if not os.path.exists(template_path):
            return "Template not found", 404

        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        uploaded_file.save(filepath)

        img_src = cv2.imread(filepath)
        faces_src = faceapp.get(img_src)
        if len(faces_src) == 0:
            return "No face detected", 400
        source_face = faces_src[0]

        cap = cv2.VideoCapture(template_path)
        temp_dir = os.path.join(TEMP_FRAMES_FOLDER, str(uuid.uuid4()))
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

        output_path = os.path.join(STATIC_FOLDER, f"output_{uuid.uuid4().hex}.mp4")
        ffmpeg_cmd = f"ffmpeg -y -framerate 25 -i {temp_dir}/frame_%05d.jpg -c:v libx264 -pix_fmt yuv420p {output_path}"
        os.system(ffmpeg_cmd)

        return send_file(output_path, mimetype='video/mp4')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
