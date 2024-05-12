from torchvision import datasets

class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, num_samples=None):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        # CIFAR-10 class names to convert indices to names
        self.class_names = self.classes
        # Use all samples if num_samples is not specified
        self.num_samples = num_samples if num_samples is not None else len(self)
    
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        target_name = self.class_names[target]
        return image, target_name

    def __len__(self):
        return min(self.num_samples, super().__len__())
