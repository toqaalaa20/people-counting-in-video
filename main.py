import cv2
import numpy as np
from ultralytics import YOLO
import argparse


# Argument parser
parser = argparse.ArgumentParser(description='Process a video with YOLO object detection.')
parser.add_argument('input_video', type=str, help='Path to the input video file')
parser.add_argument('output_video', type=str, help='Path to the output video file')
args = parser.parse_args()

# Paths
vid_path = args.input_video
output_path = args.output_video


# Load YOLO model
model = YOLO("yolov9s.pt")

# Define the polygon zone
polygon = np.array([[446, 515], [802, 287], [946, 791], [706, 1055], [542, 1031]])

# Function to check if a point is inside a polygon
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Function to draw bounding boxes and polygon, and add people count
def draw_annotations(frame, detections, polygon):
    count = 0
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = map(int, detection[:6])
        if class_id == 0:  # Only consider class_id == 0 (person)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if point_in_polygon(center, polygon):
                count += 1
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    
    # Draw polygon
    cv2.polylines(frame, [polygon], isClosed=True, color=(255, 255, 255), thickness=6)

    # Add people count on the polygon
    text = f"Count: {count}"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = polygon[0][0] - text_size[0] // 2
    text_y = polygon[0][1] - text_size[1] // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

# Open video
cap = cv2.VideoCapture(vid_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame, imgsz=1280)[0]
    detections = results.boxes.data.cpu().numpy()  # Convert to numpy array

    # Draw annotations
    frame = draw_annotations(frame, detections, polygon)

    # Write the frame to the output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
