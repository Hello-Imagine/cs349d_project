import sys
import torch
import numpy as np
# import cv2
from pathlib import Path

sys.path.append('submodules/yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression

class YOLOv5SegmentationDetector:
    def __init__(self, weights_path, img_size=640, conf_thresh=0.25, iou_thresh=0.45):
        # Initialize the device to run the model on
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the model
        self.model = DetectMultiBackend(weights_path)
        self.stride = self.model.stride
        self.names = self.model.names
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Load model into the specified device and set it to evaluation mode
        self.model.to(self.device).eval()

    def detect(self, imgs, query_item=None):
        # Input imgs shape [B, C, H, W]
        imgs = imgs.to(self.device)
        selected_num = 0

        # Run inference
        with torch.no_grad():
            output = self.model(imgs, augment=False, visualize=False)

        # Apply non-max suppression
        pred = non_max_suppression(output[0], self.conf_thresh, self.iou_thresh, classes=None, agnostic=False)

        # Process detections
        all_detected = []
        
        for det in pred:
            detected_objects = set()

            # If there are detections in the image
            if det is not None and len(det):
                for *_, conf, cls in det:
                    # Filter out detections with low confidence
                    if conf < self.conf_thresh or not (0<=int(cls)<80): continue

                    detected_objects.add(self.names[int(cls)])
            
            # Count the number of selected items
            if query_item is not None and query_item in detected_objects:
                selected_num += 1
            # Append the detected object to the list
            all_detected.append(list(detected_objects))

        return all_detected, selected_num


