from torchvision.transforms import Compose, Resize, ToTensor

from dataloader.utils import make_square
from dataloader.voc_dataloader import VOCDataLoader
from ml_udf.yolov5.yolov5_detector import YOLOv5Detector

VOC_IMAGE_DIR = 'data/VOC2012/JPEGImages'
VOC_ANNOTATION_DIR = 'data/VOC2012/Annotations'

def main():
    transform = Compose([
        Resize((640, 640)),
        ToTensor()
    ])
    # Setup data loader
    data_loader = VOCDataLoader(VOC_IMAGE_DIR, VOC_ANNOTATION_DIR, transform=transform).get_data_loader()

    # Setup YOLO detector
    detector = YOLOv5Detector()

    # Run inference
    for i, batch in enumerate(data_loader):
        images = batch['image']  # Extract images from batch
        detected_objects = detector.detect(images)
        print(f"Batch {i + 1}:")
        print("Detected objects:", detected_objects)

if __name__ == '__main__':
    main()