import os
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, annotation_file, image_dir, category_names=None, num_images=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.category_ids = None
        if category_names:
            self.category_ids = self.coco.getCatIds(catNms=category_names)
        self.image_ids = self.coco.getImgIds(catIds=self.category_ids)[:num_images]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        return image_tensor, image_info
