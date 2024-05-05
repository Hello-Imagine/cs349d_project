import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data.dataloader import default_collate


class VOCDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the JPEG images.
            annotation_dir (string): Directory with all the XML annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.images = [os.path.splitext(file)[0] for file in os.listdir(image_dir) if file.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx] + '.jpg')
        annotation_name = os.path.join(self.annotation_dir, self.images[idx] + '.xml')

        # Process the image
        img = Image.open(img_name).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # Parse the XML file
        tree = ET.parse(annotation_name)
        root = tree.getroot()

        # Extract each object
        boxes = []
        labels = []
        for member in root.findall('object'):
            labels.append(member.find('name').text)

            bndbox = member.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        sample = {'image': img, 'boxes': boxes, 'labels': labels}

        return sample

def collate_fn(batch):
    """
    Custom collate function for handling batches of images, boxes, and labels.
    """
    # Separate batch data by content
    batch_images = [item['image'] for item in batch]
    batch_boxes = [item['boxes'] for item in batch]
    batch_labels = [item['labels'] for item in batch]
    # images = [item['image'] for item in batch]

    # Use default_collate to stack images into a single tensor
    images = default_collate(batch_images)

    # For boxes and labels, we just return the list of tensors
    return {
        'image': images,
        'boxes': batch_boxes,
        'labels': batch_labels
    }

class VOCDataLoader:
    def __init__(self, image_dir, annotation_dir, batch_size=4, shuffle=True, num_workers=1, transform=None):
        self.dataset = VOCDataset(image_dir, annotation_dir, transform=transform)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, collate_fn=collate_fn)

    def get_data_loader(self):
        return self.data_loader