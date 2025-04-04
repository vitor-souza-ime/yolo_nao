# NAO Object Detection with YOLOv8

This project allows you to use the NAO robot for real-time object detection using YOLOv8, combined with the robot's camera to detect objects and announce them verbally. The object detection results are displayed using OpenCV, and the robot will announce the objects it detects.

## Features

- Real-Time Object Detection: Utilizes YOLOv8 to detect objects in images captured by the NAO robot’s camera.
- Object Announcement: NAO announces the detected objects using its Text-to-Speech (TTS) system.
- Continuous Detection Mode: Periodically captures images from the camera, detects objects, and announces the results.

## Requirements

Before running the script, make sure you have the following dependencies installed:

1. Python Libraries:
   - qi (for NAO robot communication)
   - argparse (for argument parsing)
   - time (for timing intervals)
   - torch (for PyTorch and YOLO model)
   - numpy (for array manipulation)
   - opencv-python (for image processing)
   - PIL (for image handling)
   - ultralytics (for YOLOv8 model handling)

2. NAO Robot:
   - Make sure you have access to the NAO robot and can connect to it using its IP and port.

3. YOLOv8 Model:
   - The script uses the YOLOv8 model for object detection. By default, it uses the pretrained model yolov8n.pt, but you can provide a custom model path.

4. NAO Robot Setup:
   - Ensure that the NAO robot's video and text-to-speech services are enabled and accessible.

## Usage

### Running the Detection Script

You can run the detection script with the following command:

python source01.py ip

## How It Works

### 1. Connecting to NAO:
   The script connects to the NAO robot over the specified IP and port. It then accesses the NAO robot's video and text-to-speech services.

### 2. Capturing Images:
   The robot's camera is used to capture images, which are then processed by the YOLOv8 model for object detection.

### 3. Object Detection with YOLOv8:
   The YOLOv8 model processes the captured images and detects objects. The model returns bounding boxes, class names, and confidence scores for each detected object.

### 4. Display and Announce Results:
   The detected objects are drawn on the image with bounding boxes and labels. The robot announces the detected objects using its text-to-speech system.

### 5. Continuous Detection:
   The detection process repeats at a specified interval, continuously capturing images, detecting objects, and announcing the results.

## Troubleshooting

### 1. Connection Issues:
   If you encounter connection issues with the NAO robot, check the following:
   - Ensure the robot is powered on and connected to the same network as your computer.
   - Verify the IP address and port number.

### 2. Camera Feed Issues:
   If the script fails to capture images from the NAO camera:
   - Ensure that the camera service is properly initialized and that the robot's camera is working.
   - Check the camera resolution and color space settings.

### 3. YOLOv8 Model Issues:
   If there are issues loading the model:
   - Ensure that the model file path is correct.
   - If using a custom model, ensure it is compatible with the YOLOv8 architecture.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to contribute or open issues if you encounter any problems or have suggestions for improvements!
