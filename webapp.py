from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import argparse
import supervision as sv

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLO Live')
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def init_camera(resolution):
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video.")
        exit()
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    return camera

def generate_frames(camera, model):
    box_annotator = sv.BoxAnnotator(thickness=1)

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = [
            f"{model.model.names[class_id]}, confidence={confidence:.2f}"
            for class_id, confidence in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app = Flask(__name__)
args = parse_arguments()
camera = init_camera(args.webcam_resolution)
model = YOLO("cv-ul-yolo-2.pt")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(camera, model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.teardown_appcontext
def cleanup(exception):
    if camera.isOpened():
        camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)
