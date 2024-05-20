import os
from torchvision.transforms import Compose, Resize, ToTensor
from dataloader.voc_dataloader import VOCDataLoader
import torchvision
import torch
import matplotlib.pyplot as plt

VOC_IMAGE_DIR = 'data/VOC2012/JPEGImages'
VOC_ANNOTATION_DIR = 'data/VOC2012/Annotations'
QUERY_ITEM = "person"

def main():
    transform = Compose([
        Resize((640, 640)),
        ToTensor()
    ])

    # 设置数据加载器
    data_loader = VOCDataLoader(VOC_IMAGE_DIR, VOC_ANNOTATION_DIR, transform=transform).get_data_loader()

    # 加载预训练的Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 计算选择性的辅助变量
    total_selected = 0

    # 运行推理
    for i, batch in enumerate(data_loader):
        images = batch['image']
        labels = batch['labels']

        # 在图像上运行目标检测
        with torch.no_grad():
            outputs = model(images)

        # 检查是否有person目标被检测到
        for j in range(len(images)):
            detected_objects = labels[j]

            if QUERY_ITEM in detected_objects:
                total_selected += 1

                # 显示包含person目标的图像
                plt.figure(figsize=(12, 8))
                plt.imshow(images[j].permute(1, 2, 0))
                ax = plt.gca()

                # 获取person目标的边界框和置信度得分
                person_indices = [i for i, label in enumerate(outputs[j]['labels']) if label == 1]
                person_boxes = outputs[j]['boxes'][person_indices].detach().numpy()
                person_scores = outputs[j]['scores'][person_indices].detach().numpy()

                # 在图像上绘制person目标的边界框和置信度得分
                for box, score in zip(person_boxes, person_scores):
                    if score > 0.5:
                        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2))
                        ax.text(box[0], box[1], f"Person: {score:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

                plt.axis('off')
                plt.show()

        print(f"Batch {i + 1} processed.")

    # 计算选择性
    selectivity = total_selected / len(data_loader.dataset)
    print(f"Selectivity: {selectivity}")

if __name__ == '__main__':
    main()