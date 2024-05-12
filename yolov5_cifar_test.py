from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader.cifar_dataloader import CustomCIFAR10
from ml_udf.yolov5.yolo_detector import YOLOv5SegmentationDetector

CIFAR_ROOT_DIR = 'data'
QUERY_ITEM = "person"

def main(num_images=20):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    dataset = CustomCIFAR10(root=CIFAR_ROOT_DIR, train=False, transform=transform, num_samples=num_images)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Setup YOLO detector
    detector = YOLOv5SegmentationDetector('yolov5s.pt', conf_thresh=0.1)

    # Helper variables to calculate selectivity
    total_selected = 0

    # Run inference
    for images, labels in data_loader:
        detected_objects, selected_num = detector.detect(images, QUERY_ITEM)
        total_selected += selected_num
        print("Detected objects names:", detected_objects)
        print("Labels:", labels)
    
    selectivity = total_selected / len(dataset)
    print(f"Selectivity: {selectivity}")

if __name__ == '__main__':
    main(num_images=16)