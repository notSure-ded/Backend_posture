from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
import os

# Add error handling for package installation
try:
    print("Installing OpenCV...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "opencv-python", "opencv-contrib-python", "opencv-python-headless", "-y"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    
    result = subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.10.0.84", "--no-cache-dir", "--force-reinstall"], 
                           capture_output=True, text=True, check=True)
    print("OpenCV installation completed")
except subprocess.CalledProcessError as e:
    print(f"Failed to install OpenCV: {e}")
    print(f"stdout: {e.stdout}")
    print(f"stderr: {e.stderr}")
    sys.exit(1)

# Import with error handling
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"Failed to import OpenCV: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    import numpy as np
    import tempfile
    import base64
    print("All imports successful")
except ImportError as e:
    print(f"Failed to import required packages: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculate angle between three 2D points."""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0

def analyze_posture_frame(landmarks):
    try:
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        right_hip = [landmarks[24].x, landmarks[24].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        right_knee = [landmarks[26].x, landmarks[26].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]
        right_ankle = [landmarks[28].x, landmarks[28].y]
        nose = [landmarks[0].x, landmarks[0].y]

        shoulder_mid = np.mean([left_shoulder, right_shoulder], axis=0)
        hip_mid = np.mean([left_hip, right_hip], axis=0)

        back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        neck_angle = calculate_angle(nose, shoulder_mid, hip_mid)

        is_sitting = abs(left_hip[1] - left_knee[1]) < 0.25
        is_squatting = abs(left_hip[1] - left_knee[1]) > 0.25

        good_posture = True
        if is_sitting:
            if neck_angle < 150 or back_angle < 150:
                good_posture = False
        elif is_squatting:
            if back_angle < 150 or left_knee[0] < left_ankle[0]:
                good_posture = False

        return good_posture, back_angle, neck_angle
    except Exception as e:
        print(f"Error analyzing posture: {e}")
        return None, None, None

def encode_frame_to_base64(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'opencv_version': cv2.__version__}), 200

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
            return jsonify({'error': 'Could not open video file'}), 400

        total_frames = good_frames = bad_frames = 0
        frames_data = []

        with mp_pose.Pose(static_image_mode=False) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if not results.pose_landmarks:
                    continue

                try:
                    landmarks = results.pose_landmarks.landmark
                    left_shoulder = [landmarks[11].x, landmarks[11].y]
                    right_shoulder = [landmarks[12].x, landmarks[12].y]
                    left_hip = [landmarks[23].x, landmarks[23].y]
                    right_hip = [landmarks[24].x, landmarks[24].y]
                    left_knee = [landmarks[25].x, landmarks[25].y]
                    right_knee = [landmarks[26].x, landmarks[26].y]
                    left_ankle = [landmarks[27].x, landmarks[27].y]
                    right_ankle = [landmarks[28].x, landmarks[28].y]
                    nose = [landmarks[0].x, landmarks[0].y]

                    shoulder_mid = np.mean([left_shoulder, right_shoulder], axis=0)
                    hip_mid = np.mean([left_hip, right_hip], axis=0)

                    back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                    neck_angle = calculate_angle(nose, shoulder_mid, hip_mid)

                    is_sitting = abs(left_hip[1] - left_knee[1]) < 0.25
                    is_squatting = abs(left_hip[1] - left_knee[1]) > 0.25

                    good_posture = True
                    if is_sitting:
                        if neck_angle < 150 or back_angle < 150:
                            good_posture = False
                    elif is_squatting:
                        if back_angle < 150 or left_knee[0] < left_ankle[0]:
                            good_posture = False

                    feedback = "✅ Good posture" if good_posture else "⚠️ Bad posture"
                    good_frames += good_posture
                    bad_frames += not good_posture

                    # Draw landmarks and convert frame
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    encoded_frame = encode_frame_to_base64(frame)

                    if encoded_frame:
                        frames_data.append({
                            'frame': f'data:image/jpeg;base64,{encoded_frame}',
                            'feedback': feedback
                        })

                except Exception as e:
                    print(f"[Frame skipped] Error: {e}")
                    continue

        cap.release()
        os.remove(path)

        posture_score = round((good_frames / total_frames) * 100, 2) if total_frames else 0

        return jsonify({
            'total_frames': total_frames,
            'good_posture_frames': good_frames,
            'bad_posture_frames': bad_frames,
            'posture_score': posture_score,
            'message': '✅ Good posture overall!' if posture_score > 80 else '⚠️ Fix your posture!',
            'frames': frames_data
        })

    except Exception as e:
        print(f"Error in analyze endpoint: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

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
            return jsonify({'error': 'Could not open video file'}), 400

        frame_results = []

        with mp_pose.Pose(static_image_mode=False) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                feedback = "⚠️ No person detected"

                if results.pose_landmarks:
                    good_posture, back_angle, neck_angle = analyze_posture_frame(results.pose_landmarks.landmark)
                    if good_posture is not None:
                        feedback = "✅ Good posture" if good_posture else "⚠️ Bad posture"

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                frame_results.append({
                    "frame": f"data:image/jpeg;base64,{frame_b64}",
                    "feedback": feedback
                })

        cap.release()
        os.remove(path)

        return jsonify(frame_results)

    except Exception as e:
        print(f"Error in analyze_frames endpoint: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
