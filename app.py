from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle
    except Exception as e:
        print(f"Angle error: {e}")
        return 0

def analyze_posture_frame(landmarks):
    try:
        ls, rs = [landmarks[11].x, landmarks[11].y], [landmarks[12].x, landmarks[12].y]
        le, re = [landmarks[13].x, landmarks[13].y], [landmarks[14].x, landmarks[14].y]
        lh, rh = [landmarks[23].x, landmarks[23].y], [landmarks[24].x, landmarks[24].y]
        nose = [landmarks[0].x, landmarks[0].y]
        
        left_back = calculate_angle(le, ls, lh)
        right_back = calculate_angle(re, rs, rh)
        neck_angle = calculate_angle(nose, [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2], [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2])
        back_angle = (left_back + right_back) / 2

        good_posture = back_angle > 150 and neck_angle < 30
        return good_posture, back_angle, neck_angle
    except Exception as e:
        print(f"Posture error: {e}")
        return None, None, None

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

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        video = request.files['video']
        if not video:
            return jsonify({'error': 'No video uploaded'}), 400

        temp_path = "/tmp/uploaded_video.mp4"
        video.save(temp_path)

        cap = cv2.VideoCapture(temp_path)
        good_frames, bad_frames, total_frames = 0, 0, 0
        result_frames = []

        with mp_pose.Pose(static_image_mode=False) as pose:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                total_frames += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    good_posture, _, _ = analyze_posture_frame(results.pose_landmarks.landmark)
                    if good_posture: good_frames += 1
                    else: bad_frames += 1
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                result_frames.append({
                    "frame": "data:image/jpeg;base64," + encode_frame_to_base64(frame)
                })

        score = int((good_frames / total_frames) * 100) if total_frames > 0 else 0
        feedback = "✅ Good posture overall" if score >= 75 else "⚠️ Needs improvement"

        return jsonify({
            "total_frames": total_frames,
            "good_frames": good_frames,
            "bad_frames": bad_frames,
            "score": score,
            "feedback": feedback,
            "frames": result_frames
        })
    except Exception as e:
        print(f"Analyze error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_frames', methods=['POST'])
def analyze_frames():
    try:
        video = request.files['video']
        if not video:
            return jsonify({'error': 'No video uploaded'}), 400

        temp_path = "/tmp/uploaded_video.mp4"
        video.save(temp_path)

        cap = cv2.VideoCapture(temp_path)
        result_frames = []

        with mp_pose.Pose(static_image_mode=False) as pose:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                feedback = "❌ No posture detected"
                if results.pose_landmarks:
                    good_posture, _, _ = analyze_posture_frame(results.pose_landmarks.landmark)
                    feedback = "✅ Good posture" if good_posture else "⚠️ Bad posture"
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                result_frames.append({
                    "frame": "data:image/jpeg;base64," + encode_frame_to_base64(frame),
                    "feedback": feedback
                })

        return jsonify({"frames": result_frames})
    except Exception as e:
        print(f"analyze_frames error: {e}")
        return jsonify({'error': str(e)}), 500

