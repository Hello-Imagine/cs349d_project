import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms
from pycocotools.coco import COCO
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 定义CocoDataset类，用于加载和预处理COCO数据集
class CocoDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        # 转换图像
        if self.transform is not None:
            img = self.transform(img)
        
        # 提取边界框和标签
        boxes = []
        labels = []
        for ann in coco_annotation:
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])  # COCO的bbox是[x, y, width, height]
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        return img, target

    def __len__(self):
        return len(self.ids)

def get_model(num_classes):
    # 加载预训练的Faster R-CNN模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 替换预测器的类别数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main():
    # 设置超参数
    batch_size = 4  # 注意，Faster R-CNN通常使用较小的批量大小
    num_epochs = 10
    learning_rate = 0.005
    
    # 检测并选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 确保数据集存在
    assert os.path.exists('data/train2017'), "数据集 train2017 不存在,请手动下载并放置在 data/ 目录下"
    assert os.path.exists('data/val2017'), "数据集 val2017 不存在,请手动下载并放置在 data/ 目录下"
    assert os.path.exists('data/annotations'), "数据集标注文件不存在,请手动下载并放置在 data/annotations/ 目录下"

    # 加载COCO数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CocoDataset(root='data/train2017', 
                                annFile='data/annotations/instances_train2017.json',
                                transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    val_dataset = CocoDataset(root='data/val2017',
                              annFile='data/annotations/instances_val2017.json', 
                              transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: tuple(zip(*x)))

    # 初始化模型和优化器
    model = get_model(2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item()}')

        # 在验证集上评估模型性能
        # 验证代码可以添加类似训练代码，确保进行适当的评估

if __name__ == '__main__':
    main()
