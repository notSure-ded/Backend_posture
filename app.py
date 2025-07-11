from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
import os

# Auto-install OpenCV
try:
    print("Installing OpenCV...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "opencv-python", "opencv-contrib-python", "opencv-python-headless", "-y"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.10.0.84", "--no-cache-dir", "--force-reinstall"],
                   check=True, capture_output=True, text=True)
    print("OpenCV installed.")
except subprocess.CalledProcessError as e:
    print(f"Failed to install OpenCV: {e}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
    sys.exit(1)

# Imports
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    import base64
    import tempfile
    print("Imports successful.")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Angle calculation
def calculate_angle(a, b, c):
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle
    except Exception as e:
        print(f"Angle error: {e}")
        return 0

# Analyze posture logic
def analyze_posture_frame(landmarks):
    try:
        ls, rs = [landmarks[11].x, landmarks[11].y], [landmarks[12].x, landmarks[12].y]
        lh, rh = [landmarks[23].x, landmarks[23].y], [landmarks[24].x, landmarks[24].y]
        lk, rk = [landmarks[25].x, landmarks[25].y], [landmarks[26].x, landmarks[26].y]
        la, ra = [landmarks[27].x, landmarks[27].y], [landmarks[28].x, landmarks[28].y]
        nose = [landmarks[0].x, landmarks[0].y]

        shoulder_mid = np.mean([ls, rs], axis=0)
        hip_mid = np.mean([lh, rh], axis=0)

        back_angle = calculate_angle(ls, lh, lk)
        neck_angle = calculate_angle(nose, shoulder_mid, hip_mid)

        is_sitting = abs(lh[1] - lk[1]) < 0.25
        is_squatting = abs(lh[1] - lk[1]) > 0.25

        good_posture = True
        if is_sitting:
            if neck_angle < 150 or back_angle < 150:
                good_posture = False
        elif is_squatting:
            if back_angle < 150 or lk[0] < la[0]:
                good_posture = False

        return good_posture, back_angle, neck_angle
    except Exception as e:
        print(f"Posture error: {e}")
        return None, None, None

# Encode frame to base64
def encode_frame_to_base64(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Encoding error: {e}")
        return None

@app.route('/')
def home():
    return 'Server is alive!'

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'opencv_version': cv2.__version__}), 200

# /analyze – Summary + annotated frames
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400

        video = request.files['video']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp:
            video.save(temp.name)
            path = temp.name

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            os.remove(path)
            return jsonify({'error': 'Could not open video'}), 400

        total, good, bad = 0, 0, 0
        frames_data = []

        with mp_pose.Pose(static_image_mode=False) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                total += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if not results.pose_landmarks:
                    continue

                landmarks = results.pose_landmarks.landmark
                good_posture, back_angle, neck_angle = analyze_posture_frame(landmarks)
                verdict = "✅ Good posture" if good_posture else "⚠️ Bad posture"

                good += good_posture
                bad += not good_posture

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                encoded = encode_frame_to_base64(frame)

                if encoded:
                    frames_data.append({
                        'frame': f"data:image/jpeg;base64,{encoded}",
                        'feedback': verdict
                    })

        cap.release()
        os.remove(path)

        score = round((good / total) * 100, 2) if total else 0

        return jsonify({
            'total_frames': total,
            'good_posture_frames': good,
            'bad_posture_frames': bad,
            'posture_score': score,
            'message': '✅ Good posture overall!' if score > 80 else '⚠️ Fix your posture!',
            'frames': frames_data
        })

    except Exception as e:
        print(f"Analyze error: {e}")
        return jsonify({'error': f'Failed: {str(e)}'}), 500

# /analyze_frames – Per-frame feedback
@app.route('/analyze_frames', methods=['POST'])
def analyze_frames():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400

        video = request.files['video']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp:
            video.save(temp.name)
            path = temp.name

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            os.remove(path)
            return jsonify({'error': 'Could not open video'}), 400

        result_frames = []

        with mp_pose.Pose(static_image_mode=False) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                feedback = "⚠️ No person detected"

                if results.pose_landmarks:
                    good_posture, back_angle, neck_angle = analyze_posture_frame(results.pose_landmarks.landmark)
                    if good_posture is not None:
                        feedback = "✅ Good posture" if good_posture else "⚠️ Bad posture"
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                _, buffer = cv2.imencode('.jpg', frame)
                encoded = base64.b64encode(buffer).decode('utf-8')

                result_frames.append({
                    "frame": f"data:image/jpeg;base64,{encoded}",
                    "feedback": feedback
                })

        cap.release()
        os.remove(path)

        return jsonify(result_frames)

    except Exception as e:
        print(f"analyze_frames error: {e}")
        return jsonify({'error': f'Failed: {str(e)}'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000)) 
    print(f"🚀 Starting Flask server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
