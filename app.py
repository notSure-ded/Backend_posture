from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import subprocess

print("Starting Flask app...")

# Force uninstall opencv-contrib-python (if mistakenly installed)
try:
    print("🔧 Checking for opencv-contrib-python...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "opencv-contrib-python"],
        check=True
    )
    print("✅ Uninstalled opencv-contrib-python successfully")
except Exception as e:
    print(f"⚠️ Failed to uninstall opencv-contrib-python: {e}")

# Optional: force reinstall correct version
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "opencv-python"],
        check=True
    )
    print("✅ Installed correct opencv-python")
except Exception as e:
    print(f"⚠️ Failed to install opencv-python: {e}")

# Prevent EXR error
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

# Imports with error handling
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    import base64
    import tempfile
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()


app = Flask(__name__)
CORS(app)

# Initialize MediaPipe only if imports succeeded
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe initialized")
except Exception as e:
    print(f"❌ MediaPipe initialization failed: {e}")
    mp_pose = None
    mp_drawing = None

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
        print(f"Angle calculation error: {e}")
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
        print(f"Posture analysis error: {e}")
        return None, None, None

# Encode frame to base64
def encode_frame_to_base64(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Frame encoding error: {e}")
        return None

@app.route('/')
def home():
    return jsonify({
        'message': 'Posture Analysis API is running!',
        'status': 'healthy',
        'endpoints': ['/health', '/analyze', '/analyze_frames']
    })

@app.route('/health', methods=['GET'])
def health():
    try:
        cv2_version = cv2.__version__ if 'cv2' in globals() else 'Not available'
        return jsonify({
            'status': 'healthy',
            'opencv_version': cv2_version,
            'mediapipe_available': 'mp' in globals(),
            'python_version': sys.version
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if MediaPipe is available
    if not mp_pose or not mp_drawing:
        return jsonify({'error': 'MediaPipe not available - system dependencies missing'}), 500
        
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video = request.files['video']
        if video.filename == '':
            return jsonify({'error': 'No video file selected'}), 400

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp:
            video.save(temp.name)
            path = temp.name

        # Process video
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            os.remove(path)
            return jsonify({'error': 'Could not open video file'}), 400

        total_frames, good_frames, bad_frames = 0, 0, 0
        frames_data = []

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                total_frames += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if not results.pose_landmarks:
                    continue

                landmarks = results.pose_landmarks.landmark
                good_posture, back_angle, neck_angle = analyze_posture_frame(landmarks)
                
                if good_posture is not None:
                    verdict = "✅ Good posture" if good_posture else "⚠️ Bad posture"
                    good_frames += int(good_posture)
                    bad_frames += int(not good_posture)
                else:
                    verdict = "❓ Could not analyze"

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                encoded_frame = encode_frame_to_base64(frame)

                if encoded_frame:
                    frames_data.append({
                        'frame': f"data:image/jpeg;base64,{encoded_frame}",
                        'feedback': verdict
                    })

        cap.release()
        os.remove(path)

        # Calculate score
        score = round((good_frames / total_frames) * 100, 2) if total_frames > 0 else 0

        return jsonify({
            'total_frames': total_frames,
            'good_posture_frames': good_frames,
            'bad_posture_frames': bad_frames,
            'posture_score': score,
            'message': '✅ Good posture overall!' if score > 80 else '⚠️ Consider improving your posture!',
            'frames': frames_data
        })

    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/analyze_frames', methods=['POST'])
def analyze_frames():
    # Check if MediaPipe is available
    if not mp_pose or not mp_drawing:
        return jsonify({'error': 'MediaPipe not available - system dependencies missing'}), 500
        
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video = request.files['video']
        if video.filename == '':
            return jsonify({'error': 'No video file selected'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp:
            video.save(temp.name)
            path = temp.name

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            os.remove(path)
            return jsonify({'error': 'Could not open video file'}), 400

        result_frames = []

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

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
        print(f"Frame analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Frame analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask server on 0.0.0.0:{port}")
    print(f"📱 Health check: http://0.0.0.0:{port}/health")
    app.run(host="0.0.0.0", port=port, debug=False)
