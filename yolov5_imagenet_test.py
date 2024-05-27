import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 将 yolov5 目录添加到系统路径
sys.path.insert(0, str(Path.cwd() / 'yolov5'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from dataloader import get_imagenet_val_loader

def run_detection(image, model, device):
    image = image.to(device)  # 转移到设备
    pred = model(image)  # 运行模型
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)  # 非极大值抑制
    return pred

def plot_results(image, pred, model, save_path=None):
    image = image[0].permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    ax = plt.gca()
    for det in pred:  # detections per image
        if len(det):
            det[:, :4] = scale_coords(image.shape[:2], det[:, :4], image.shape[:2]).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=2)
    if save_path:
        plt.savefig(save_path)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 配置参数
    root_dir = '/Users/yangliuxin/Desktop/CS349D/Project/cs349d_project/data/imagenet/ILSVRC2012_img_val 11.44.52'
    weights = 'yolov5s.pt'  # 使用预训练的 YOLOv5 模型权重
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # 选择设备

    # 加载数据
    val_loader = get_imagenet_val_loader(root_dir)

    # 加载模型
    model = DetectMultiBackend(weights=weights, device=device)
    model.eval()

    # 遍历验证集图像
    for i, (image, img_path) in enumerate(val_loader):
        pred = run_detection(image, model, device)
        plot_results(image, pred, model, save_path=f'results/{Path(img_path[0]).name}')
