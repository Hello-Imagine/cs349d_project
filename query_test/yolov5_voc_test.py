from torchvision.transforms import Compose, Resize, ToTensor

from dataloader.voc_dataloader import VOCDataLoader
from ml_udf.yolov5.yolo_detector import YOLOv5SegmentationDetector
from config import QUERY_ITEM, VOC_IMAGE_DIR, VOC_ANNOTATION_DIR

def main():
    transform = Compose([
        Resize((640, 640)),
        ToTensor()
    ])
    
    # Setup data loader
    data_loader = VOCDataLoader(VOC_IMAGE_DIR, VOC_ANNOTATION_DIR, transform=transform).get_data_loader()

    # Setup YOLO detector
    detector = YOLOv5SegmentationDetector('yolov5s.pt')

    # Helper variables to calculate selectivity
    total_selected = 0

    # Run inference
    for i, batch in enumerate(data_loader):
        images = batch['image']  # Extract images from batch
        detected_objects, selected_num = detector.detect(images, QUERY_ITEM)
        total_selected += selected_num

        print(f"Batch {i + 1}:")
        print("Detected objects names:", detected_objects)
    
    selectivity = total_selected / len(data_loader.dataset)
    print(f"Selectivity: {selectivity}")

if __name__ == '__main__':
    main()