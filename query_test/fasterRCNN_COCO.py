import torch
import torchvision
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt

from config import QUERY_ITEM

def run_detection(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 将图像转换为PyTorch tensor
    image_tensor = transform(image).unsqueeze(0)
    # 在图像上运行模型
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs

if __name__ == "__main__":
    # 加载COCO验证集
    coco = COCO('dataCOCO/annotations/instances_val2017.json')
    
    # 加载预训练的Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # 获取所有图像ID
    image_ids = coco.getImgIds()
    
    # 选择前20个图像ID
    num_images = 20
    selected_image_ids = image_ids[:num_images]
    
    # 获取QUERY_ITEM类的ID
    query_category_id = coco.getCatIds(catNms=[QUERY_ITEM])[0]
    
    # 初始化计数器
    num_images_with_query = 0
    
    # 遍历选定的图像
    for image_id in selected_image_ids:
        # 加载图像
        image_info = coco.loadImgs(image_id)[0]
        image_path = f"data/val2017/{image_info['file_name']}"
        image = Image.open(image_path).convert('RGB')
        
        # 在图像上运行目标检测
        outputs = run_detection(image)
        
        # 检查是否有QUERY_ITEM目标被检测到
        detected_category_ids = outputs[0]['labels'].numpy()
        if query_category_id in detected_category_ids:
            num_images_with_query += 1
            
            # 显示包含QUERY_ITEM目标的图像
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            ax = plt.gca()
            
            # 获取QUERY_ITEM目标的边界框和置信度得分
            query_boxes = outputs[0]['boxes'][outputs[0]['labels'] == query_category_id].numpy()
            query_scores = outputs[0]['scores'][outputs[0]['labels'] == query_category_id].numpy()
            
            # 在图像上绘制QUERY_ITEM目标的边界框和置信度得分
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