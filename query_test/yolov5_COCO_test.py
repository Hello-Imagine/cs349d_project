import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO

from config import QUERY_ITEM

# 加载YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def run_detection(image):
    # 在图像上运行YOLOv5模型
    results = model(image)
    return results

if __name__ == "__main__":
    # 加载COCO验证集
    coco = COCO('dataCOCO/annotations/instances_val2017.json')

    # 获取所有图像ID
    image_ids = coco.getImgIds()

    # 选择前20个图像ID
    num_images = 20
    selected_image_ids = image_ids[:num_images]

    # 获取 QUERY_ITEM 类的ID
    category_id = coco.getCatIds(catNms=[QUERY_ITEM])[0]

    # 初始化计数器
    num_images_with_query = 0

    # 遍历选定的图像
    for image_id in selected_image_ids:
        # 加载图像
        image_info = coco.loadImgs(image_id)[0]
        image_path = f"dataCOCO/val2017/{image_info['file_name']}"
        image = Image.open(image_path).convert('RGB')

        # 在图像上运行目标检测
        results = run_detection(image)

        # 检查是否有query目标被检测到
        detected_category_ids = results.xyxy[0][:, 5].numpy().astype(int)
        if category_id in detected_category_ids:
            num_images_with_query += 1

            # 显示包含query目标的图像
            plt.figure(figsize=(12, 8))
            plt.imshow(np.array(image))
            ax = plt.gca()

            # 获取query目标的边界框和置信度得分
            query_boxes = results.xyxy[0][results.xyxy[0][:, 5] == category_id][:, :4].numpy()
            query_scores = results.xyxy[0][results.xyxy[0][:, 5] == category_id][:, 4].numpy()

            # 在图像上绘制query目标的边界框和置信度得分
            for box, score in zip(query_boxes, query_scores):
                if score > 0.5:
                    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                               fill=False, edgecolor='red', linewidth=2))
                    ax.text(box[0], box[1], f"Person: {score:.2f}", fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.5))

            plt.axis('off')
            plt.show()

    # 计算selectivity
    selectivity = num_images_with_query / num_images
    print(f"Selectivity for {QUERY_ITEM} category: {selectivity:.4f}")