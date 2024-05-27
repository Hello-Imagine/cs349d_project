import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

if __name__ == "__main__":
    # 加载ImageNet验证集
    imagenet_val = DatasetFolder(
        root='/Users/yangliuxin/Desktop/CS349D/Project/cs349d_project/data/imagenet/ILSVRC2012_img_val 11.44.52',
        loader=pil_loader,
        extensions=('.jpg',),
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    val_loader = DataLoader(imagenet_val, batch_size=1, shuffle=False)

    # 加载预训练的Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # 选择要计算selectivity的类别ID
    category_name = 'person'
    category_id = imagenet_val.class_to_idx[category_name]

    # 初始化计数器
    num_images = len(val_loader)
    num_images_with_category = 0

    # 遍历验证集图像
    for images, _ in val_loader:
        image = images[0]
        
        # 在图像上运行目标检测
        outputs = run_detection(image)
        
        # 检查是否有指定类别的目标被检测到
        detected_category_ids = outputs[0]['labels'].numpy()
        if category_id in detected_category_ids:
            num_images_with_category += 1
        
        # 显示包含指定类别目标的图像
        if num_images_with_category > 0 and num_images_with_category <= 5:
            plt.figure(figsize=(12, 8))
            plt.imshow(image.permute(1, 2, 0))
            ax = plt.gca()
            
            # 获取指定类别目标的边界框和置信度得分
            category_boxes = outputs[0]['boxes'][outputs[0]['labels'] == category_id].numpy()
            category_scores = outputs[0]['scores'][outputs[0]['labels'] == category_id].numpy()
            
            # 在图像上绘制指定类别目标的边界框和置信度得分
            for box, score in zip(category_boxes, category_scores):
                if score > 0.5:
                    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red', linewidth=2))
                    ax.text(box[0], box[1], f"{category_name}: {score:.2f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            plt.axis('off')
            plt.show()
    
    # 计算selectivity
    selectivity = num_images_with_category / num_images
    print(f"Selectivity for '{category_name}' category: {selectivity:.4f}")
