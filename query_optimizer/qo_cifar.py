import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader.cifar_dataloader import CustomCIFAR10
from query_optimizer.qo import QueryOptimizer

class QueryOptimizerCIFAR(QueryOptimizer):
    def setup_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CustomCIFAR10(root='data', train=False, transform=transform)
        return DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    def load_pp_model(self):
        pp_model = DNN()  # Assume DNN is defined somewhere
        pp_model.load_state_dict(torch.load('models/cifar.pt'))
        pp_model.eval()
        return pp_model
