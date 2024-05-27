import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class ImageNetValidationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.JPEG')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

def get_imagenet_val_loader(root_dir, batch_size=1, shuffle=False):
    data_transform = transforms.Compose([
        transforms.Resize((640, 640)),  # YOLOv5 expects 640x640 images
        transforms.ToTensor(),
    ])
    dataset = ImageNetValidationDataset(root_dir, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
