import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.cifar_dataloader import CustomCIFAR10
import matplotlib.pyplot as plt

from config import CIFAR_ROOT_DIR, QUERY_ITEM

# COCO数据集的类名列表
COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def main(num_images=20):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载CIFAR-10数据集
    dataset = CustomCIFAR10(root=CIFAR_ROOT_DIR, train=False, transform=transform, num_samples=num_images)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # 加载预训练的Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 计算选择性的辅助变量
    total_selected = 0

    # 运行推理
    for images, labels in data_loader:
        # 在图像上运行目标检测
        with torch.no_grad():
            outputs = model(images)

        # 检查是否有person目标被检测到
        detected_objects = []
        for i, output in enumerate(outputs):
            # 将预测标签映射到COCO数据集的类名
            detected_labels = [COCO_NAMES[label] for label in output['labels'].tolist()]
            detected_objects.extend(detected_labels)

            if QUERY_ITEM in detected_labels:
                total_selected += 1

                # 显示包含person目标的图像
                plt.figure(figsize=(12, 8))
                plt.imshow(images[i].permute(1, 2, 0))
                ax = plt.gca()

                # 获取person目标的边界框和置信度得分
                person_indices = [j for j, label in enumerate(output['labels']) if COCO_NAMES[label] == QUERY_ITEM]
                person_boxes = output['boxes'][person_indices].detach().numpy()
                person_scores = output['scores'][person_indices].detach().numpy()

                # 在图像上绘制person目标的边界框和置信度得分
                for box, score in zip(person_boxes, person_scores):
                    if score > 0.5:
                        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2))
                        ax.text(box[0], box[1], f"Person: {score:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

                plt.axis('off')
                plt.show()

        print("Detected objects names:", detected_objects)
        print("Labels:", labels)

    selectivity = total_selected / len(dataset)
    print(f"Selectivity: {selectivity}")

if __name__ == '__main__':
    main(num_images=16)