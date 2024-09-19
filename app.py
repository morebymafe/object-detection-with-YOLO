from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import argparse
import supervision as sv

model = YOLO("yolov8n.pt")

# Function on the video frame
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLO Live')
    parser.add_argument(
        "--webcam-resolution",
        default=[1200, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

# Main function
app=Flask(__name__)
args = parse_arguments()
frame_width, frame_height = args.webcam_resolution

camera=cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

box_annotator = sv.BoxAnnotator(
    # color= 'Blue',
    thickness=2,
),


def generate_frames():
    while True:
        ret, frame=camera.read()
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        frame = box_annotator.annotate(scene=frame, detections=detections)

        cv2.imshow('yolov8', frame)

        if not ret:
            break
        else:
            ret, buffer=cv2.imencode('.jpg', frame)
            frame=buffer.tobytes()
        
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template('/index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)