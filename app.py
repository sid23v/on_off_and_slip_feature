# app.py
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify, request
import threading
import time

app = Flask(__name__)


CAMERA_INDEX = 1  
FRAME_WIDTH = None  


DP = 1.75
MIN_DIST = 500
PARAM1 = 100
PARAM2 = 50
MIN_RADIUS = 30
MAX_RADIUS = 1200


CENTER_TOLERANCE = 40
RADIUS_TOLERANCE = 30


COLOR_STORED = (255, 0, 0)       
COLOR_LIVE = (0, 165, 255)       


stored_circle = None   
last_circles = None    
last_frame_time = 0.0
state_lock = threading.Lock()


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if FRAME_WIDTH is not None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open camera with index {CAMERA_INDEX}")

def generate_frames():
    global stored_circle, last_circles, last_frame_time
    while True:
        ret, frame = cap.read()
        if not ret:
            
            time.sleep(0.05)
            continue

        
        overlay = frame.copy()
        output = frame.copy()

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

        
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=DP,
            minDist=MIN_DIST,
            param1=PARAM1,
            param2=PARAM2,
            minRadius=MIN_RADIUS,
            maxRadius=MAX_RADIUS
        )

        live_circle = None

        
        with state_lock:
            if circles is not None:
                
                circles_arr = np.round(circles[0, :]).astype("int")
                last_circles = circles_arr
            else:
                last_circles = None
            last_frame_time = time.time()

        
        with state_lock:
            scopy = stored_circle  
        if scopy is not None:
            sx, sy, sr = scopy
            cv2.circle(output, (sx, sy), sr, COLOR_STORED, 3)
            cv2.circle(output, (sx, sy), 4, COLOR_STORED, -1)

        
        if circles is not None:
            circles_arr = np.round(circles[0, :]).astype("int")
            x, y, r = max(circles_arr, key=lambda c: c[2])
            live_circle = (x, y, r)
            cv2.circle(output, (x, y), r, COLOR_LIVE, 3)
            cv2.circle(output, (x, y), 5, COLOR_LIVE, -1)

        
        with state_lock:
            scopy = stored_circle
        if scopy is not None and live_circle is not None:
            lx, ly, lr = live_circle
            sx, sy, sr = scopy
            center_dist = np.sqrt((lx - sx)**2 + (ly - sy)**2)
            radius_diff = abs(lr - sr)
            aligned = (center_dist <= CENTER_TOLERANCE and radius_diff <= RADIUS_TOLERANCE)
            if aligned:
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
                alpha = 0.20
            else:
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
                alpha = 0.20
            output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

        
        ret2, buffer = cv2.imencode('.jpg', output)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Endoscopy Live Feed - Circle Detection</title>
  <style>
    body { font-family: Arial, sans-serif; background:#111; color:#eee; text-align:center; }
    .controls { margin-top: 10px; }
    button { font-size:16px; padding:10px 18px; margin:6px; border-radius:6px; cursor:pointer; }
    #status { margin-top:8px; color:#ddd; }
    img#video { border: 4px solid #222; max-width: 90vw; height: auto; }
  </style>
</head>
<body>
  <h2>Endoscopy Live Feed - Circle Detection</h2>
  <img id="video" src="{{ url_for('video_feed') }}" alt="video feed"/>
  <div class="controls">
    <button id="setBtn">Set Circle (S)</button>
    <button id="resetBtn">Reset Circle (R)</button>
  </div>
  <div id="status">Status: Ready</div>

<script>
  const setBtn = document.getElementById('setBtn');
  const resetBtn = document.getElementById('resetBtn');
  const statusDiv = document.getElementById('status');

  async function postJSON(url) {
    try {
      const resp = await fetch(url, { method: 'POST' });
      return resp.json();
    } catch (err) {
      return { ok:false, message: String(err) };
    }
  }

  setBtn.onclick = async () => {
    statusDiv.textContent = 'Status: Setting circle...';
    const data = await postJSON('/set_circle');
    if (data.ok) {
      statusDiv.textContent = 'Status: ' + data.message;
    } else {
      statusDiv.textContent = 'Status: ' + data.message;
    }
  };

  resetBtn.onclick = async () => {
    statusDiv.textContent = 'Status: Resetting circle...';
    const data = await postJSON('/reset_circle');
    statusDiv.textContent = 'Status: ' + (data.ok ? data.message : data.message);
  };

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === 's' || e.key === 'S') {
      setBtn.click();
    } else if (e.key === 'r' || e.key === 'R') {
      resetBtn.click();
    }
  });
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_circle', methods=['POST'])
def set_circle():
    """Average last detected circles and store the averaged circle."""
    global stored_circle, last_circles
    with state_lock:
        lc = last_circles.copy() if last_circles is not None else None

    if lc is None:
        return jsonify(ok=False, message="No circles detected in the last frame. Try again.")

    try:
        avg_x = int(np.mean(lc[:, 0]))
        avg_y = int(np.mean(lc[:, 1]))
        avg_r = int(np.mean(lc[:, 2]))
    except Exception as e:
        return jsonify(ok=False, message=f"Error averaging circles: {e}")

    with state_lock:
        stored_circle = (avg_x, avg_y, avg_r)

    return jsonify(ok=True, message=f"SET circle - Center: ({avg_x}, {avg_y}), Radius: {avg_r}")

@app.route('/reset_circle', methods=['POST'])
def reset_circle():
    global stored_circle
    with state_lock:
        stored_circle = None
    return jsonify(ok=True, message="RESET circle")

def cleanup():
    try:
        cap.release()
    except:
        pass

import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
