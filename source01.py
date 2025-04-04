import qi
import argparse
import time
import torch
import numpy as np
import cv2
import traceback
from PIL import Image
from ultralytics import YOLO


class NAOObjectDetector:
    def __init__(self, ip, port, model_path=None):
        # Connect to NAO robot
        self.ip = ip
        self.port = port
        self.session = None
        self.video_client = None
        self.camera_service = None
        self.tts_service = None
        self.connect_to_nao()
        
        # Load the YOLOv8 model
        try:
            print("Loading YOLOv8 model...")
            if model_path:
                self.model = YOLO(model_path)
            else:                
                self.model = YOLO("yolov8n.pt")  
            print("YOLOv8 model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            traceback.print_exc()
            raise
        
        # Create a CV2 window for displaying results
        cv2.namedWindow("NAO Object Detection (YOLOv8)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("NAO Object Detection (YOLOv8)", 800, 600)
    
    def connect_to_nao(self):
        """Connect to the NAO robot and initialize services"""
        try:
            self.session = qi.Session()
            self.session.connect(f"tcp://{self.ip}:{self.port}")
            print(f"Connected to NAO at {self.ip}:{self.port}")
            
            # Get the camera service
            self.camera_service = self.session.service("ALVideoDevice")
            self.tts_service = self.session.service("ALTextToSpeech")
            
            # Initialize the video client
            self.resolution = 2  # VGA: 640x480
            self.colorSpace = 11  # RGB
            self.fps = 30
            self.camera_index = 0  # Top camera
            
            # Unsubscribe any existing instances to avoid resource conflicts
            try:
                self.camera_service.unsubscribe("NAO_YOLO_Detection")
            except:
                pass
                
            # Subscribe to the camera feed
            self.video_client = self.camera_service.subscribeCamera(
                "NAO_YOLO_Detection", self.camera_index, 
                self.resolution, self.colorSpace, self.fps
            )
            
            return True
        except Exception as e:
            print(f"Error connecting to NAO: {e}")
            traceback.print_exc()
            return False
    
    def capture_image(self):
        """Capture an image from NAO's camera"""
        try:
            if not self.camera_service or not self.video_client:
                print("Camera service not initialized properly")
                return None
                
            nao_image = self.camera_service.getImageRemote(self.video_client)
            
            if nao_image is None:
                print("Cannot capture image from NAO camera")
                return None
                
            # Check if the image has the expected format
            if len(nao_image) < 7:
                print(f"Unexpected image format: {nao_image}")
                return None
                
            # Get the image data
            width = nao_image[0]
            height = nao_image[1]
            array = nao_image[6]
            
            # Create a PIL Image from the byte array
            img_str = bytes(bytearray(array))
            img = Image.frombytes("RGB", (width, height), img_str)
            
            # Convert PIL image to numpy array for YOLO
            cv_image = np.array(img)
            cv_image = cv_image[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV
            
            return cv_image
        except Exception as e:
            print(f"Error capturing image: {e}")
            traceback.print_exc()
            # Try to reconnect
            print("Attempting to reconnect to camera...")
            self.cleanup()
            time.sleep(1)
            self.connect_to_nao()
            return None
    
    def detect_objects(self, image):        
        try:            
            results = self.model(image, conf=0.4)  # Set confidence threshold
            
            # Process results
            processed_results = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    if cls < len(self.model.names):
                        class_name = self.model.names[cls]
                    else:
                        class_name = "unknown"
                    
                    processed_results.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': float(conf),
                        'class': cls,
                        'name': class_name
                    })
            
            return processed_results
        except Exception as e:
            print(f"Error in object detection: {e}")
            traceback.print_exc()
            return []
    
    def display_results(self, image, results):
        """Display the image with bounding boxes and labels"""
        try:
            # Make a copy of the image to draw on
            output_image = image.copy()
            
            detected_objects = []
            
            # Draw bounding boxes and labels on the image
            for result in results:
                box = result['box']
                name = result['name']
                conf = result['conf']
                
                # Draw rectangle
                cv2.rectangle(output_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Put text
                label_text = f'{name}: {conf:.2f}'
                cv2.putText(output_image, label_text, (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detected_objects.append(f"{name}: {conf:.2f}")
            
            # Print detected objects to console
            print("\nObjects detected:")
            if detected_objects:
                for obj in detected_objects:
                    print(obj)
            else:
                print("None")
            
            # Display the image with OpenCV
            cv2.imshow("NAO Object Detection (YOLOv8)", output_image)
            cv2.waitKey(1)  # Update the window
            
            # Save the result image with timestamp
            timestamp = int(time.time())
            cv2.imwrite(f"nao_yolo_detection_{timestamp}.png", output_image)
            
            # Speak detected objects in English
            if self.tts_service:
                self.announce_detections(detected_objects)
            
            return detected_objects
        except Exception as e:
            print(f"Error displaying results: {e}")
            traceback.print_exc()
            return []
    
    def announce_detections(self, detected_objects):
        """Make NAO announce what it has detected"""
        try:
            if not detected_objects:
                self.tts_service.say("I don't see any objects.")
                return
                
            # Get unique object names (remove duplicates)
            unique_objects = {}
            for detection in detected_objects:
                object_name = detection.split(":")[0].strip()
                if object_name in unique_objects:
                    unique_objects[object_name] += 1
                else:
                    unique_objects[object_name] = 1
            
            # Build announcement text
            announcement = "I can see "
            
            if len(unique_objects) == 1:
                object_name, count = list(unique_objects.items())[0]
                if count == 1:
                    announcement += f"a {object_name}."
                else:
                    announcement += f"{count} {object_name}s."
            else:
                object_entries = list(unique_objects.items())
                for i, (object_name, count) in enumerate(object_entries):
                    if i == len(object_entries) - 1:
                        if count == 1:
                            announcement += f"and a {object_name}."
                        else:
                            announcement += f"and {count} {object_name}s."
                    else:
                        if count == 1:
                            announcement += f"a {object_name}, "
                        else:
                            announcement += f"{count} {object_name}s, "
            
            self.tts_service.say(announcement)
        except Exception as e:
            print(f"Error announcing detections: {e}")
    
    def run_continuous_detection(self, interval=5):
        """Run continuous object detection with specified interval"""
        try:
            print("Starting continuous object detection with YOLOv8. Press Ctrl+C to stop.")
            detection_count = 0
            
            while True:
                detection_count += 1
                print(f"\n--- Detection #{detection_count} ---")
                
                # Capture image from NAO camera
                print("Capturing image from NAO camera...")
                image = self.capture_image()
                
                if image is None:
                    print("Failed to capture image, retrying...")
                    time.sleep(1)
                    continue
                    
                # Perform object detection
                print("Detecting objects using YOLOv8...")
                results = self.detect_objects(image)
                
                # Display and announce results
                detected_objects = self.display_results(image, results)
                
                # Print summary
                if detected_objects:
                    print("\nSummary of detection:")
                    print(f"Total objects detected: {len(detected_objects)}")
                    print("Object list:", ", ".join([item.split(":")[0] for item in detected_objects]))
                else:
                    print("No objects were detected.")
                
                # Wait before the next detection
                print(f"Waiting {interval} seconds before next detection...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nDetection stopped by user.")
        except Exception as e:
            print(f"Error during continuous detection: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def fine_tune_model(self, data_yaml_path, epochs=10, batch_size=16):
        """Fine-tune the YOLOv8 model on custom dataset"""
        try:
            print(f"Fine-tuning YOLOv8 model on custom dataset: {data_yaml_path}")
            
            # Set training parameters
            model_params = {
                'data': data_yaml_path,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': 640,
                'device': 'cpu',  # Use 'cpu' for compatibility or '0' for GPU if available
                'workers': 4
            }
            
            # Fine-tune the model
            results = self.model.train(**model_params)
            
            # Save the fine-tuned model
            self.model = YOLO(f'{self.model.name}_trained')
            
            print("Fine-tuning completed successfully!")
            return True
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.camera_service and self.video_client:
                self.camera_service.unsubscribe(self.video_client)
                print("Unsubscribed from NAO camera")
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NAO Object Detection with YOLOv8')
    parser.add_argument('--ip', type=str, default='10.0.1.103', help='NAO robot IP address')
    parser.add_argument('--port', type=int, default=9559, help='NAO robot port')
    parser.add_argument('--model', type=str, default=None, help='Path to custom YOLOv8 model')
    parser.add_argument('--interval', type=float, default=3.0, help='Detection interval in seconds')
    parser.add_argument('--finetune', type=str, default=None, help='Path to data.yaml for fine-tuning')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for fine-tuning')
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = NAOObjectDetector(args.ip, args.port, args.model)
    
    # Fine-tune the model if requested
    if args.finetune:
        detector.fine_tune_model(args.finetune, args.epochs)
    
    # Run continuous detection with the specified interval
    detector.run_continuous_detection(interval=args.interval)
