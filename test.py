import cv2
from ultralytics import YOLO

# Load model YOLOv8 pre-trained
model = YOLO("yolov8n.pt")

# Load the image using OpenCV
image = cv2.imread("D:/g2.jpg")

# Ensure the image was loaded correctly
if image is None:
    print("Error: Unable to load image.")
else:
    # Get results from YOLOv8
    results = model(image)

    # Hiển thị kết quả (Display results)
    for result in results:
        boxes = result.boxes.xyxy  # Get bounding box coordinates
        labels = result.boxes.cls  # Get the class labels of the detected objects
        names = result.names  # Get the class names for the labels

        for box, label in zip(boxes, labels):
            label_name = names[int(label)]  # Get the name of the label
            x1, y1, x2, y2 = map(int, box)
            # Draw rectangle around the detected object
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Display the label (name of the object)
            cv2.putText(image, label_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Resize the image to fit the display window size (optional, change 800 as needed)
    window_width = 800
    aspect_ratio = image.shape[1] / image.shape[0]
    new_height = int(window_width / aspect_ratio)
    resized_image = cv2.resize(image, (window_width, new_height))

    # Show the resized image with detections
    cv2.imshow("Object Detection", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



















