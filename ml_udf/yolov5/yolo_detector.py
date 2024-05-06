import sys
import torch
import numpy as np
import cv2
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

    def detect(self, imgs):
        # Ensure imgs is a batch of tensors [B, C, H, W] and is on the correct device
        imgs = imgs.to(self.device)

        # Inference
        with torch.no_grad():
            pred = self.model(imgs, augment=False, visualize=False)

        # Apply non-max suppression to each prediction
        preds = [non_max_suppression(p, self.conf_thresh, self.iou_thresh, classes=None, agnostic=False) for p in pred]

        # Process detections
        detected_objects = []
        for pred in preds:
            for det in pred:
                if det is not None and len(det):
                    for *xyxy, conf, cls in det:
                        # Filter out detections with low confidence
                        if conf < self.conf_thresh or not (0<=int(cls)<80): continue
                        # Rescale coordinates to original image size
                        xyxy = det[:, :4].cpu().numpy()
                        xyxy = (xyxy / self.stride).astype(int)
                        # Append the detected object to the list
                        detected_objects.append({
                            'box': xyxy,
                            'confidence': float(conf),
                            'class': self.names[int(cls)]
                        })

        return detected_objects

