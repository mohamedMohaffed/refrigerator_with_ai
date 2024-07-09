from ultralytics import YOLO
import cv2
import math

class_names = ['bottle', 'bottles', 'box', 'can', 'carton', 'container', 'cup', 'packet', 'shelf', 'void']
colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
    (255, 165, 0)   # Orange
]

# Initialize YOLO model
model = YOLO("forcoca1.pt")

# Open video file
video_path = 0 # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    res = model(frame)

    # Initialize a dictionary to count the occurrences of each class
    class_counts = {class_name: 0 for class_name in class_names}

    # Draw rectangles on the frame
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + w // 2), (y1 + h // 2)
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Correctly use class names to match and draw rectangles with appropriate colors
            class_name = class_names[cls]
            class_counts[class_name] += 1

            color = colors[cls % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the class counts on the frame
    y_offset = 30
    for class_name, count in class_counts.items():
        text = f"{class_name}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
