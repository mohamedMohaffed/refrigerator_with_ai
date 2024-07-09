from ultralytics import YOLO
import cv2
import os
import math
class_names = ['bottle', 'bottles', 'box', 'can', 'carton', 'container', 'cup', 'packet', 'shelf', 'void']


# Function to calculate area in square centimeters
def calculate_area_cm(w, h, pixels_per_cm):
    area_cm2 = (w / pixels_per_cm) * (h / pixels_per_cm)
    return area_cm2
pixels_per_cm = 10
folder = "coca-img"
image_files = os.listdir(folder)

# Initialize YOLO model outside the loop to avoid reinitializing it for every image
model = YOLO("forcoca1.pt")

for img_file in image_files:
    img_path = os.path.join(folder, img_file)
    total_shelf_area_cm2 = 0
    img = cv2.imread(img_path)

    # Perform object detection
    res = model(img)
    # Draw rectangles on the image
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + w // 2), (y1 + h // 2)
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100


            if class_names[cls] !='shelf':
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(img, f"{class_names[cls]}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                area_cm2 = calculate_area_cm(w, h, pixels_per_cm)
                cv2.putText(img, f"{class_names[cls]}: {conf:.2f}, Area: {area_cm2:.2f} cm^2", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                total_shelf_area_cm2 += area_cm2
        if total_shelf_area_cm2 !=0:
            print(f"Total area of all shelves: {total_shelf_area_cm2:.2f} cm^2")
        else:
            print("all good")

    cv2.imshow('Object Detection', img)
    cv2.waitKey(0)


cv2.destroyAllWindows()
