from torchvision import transforms
from torch.utils.data import DataLoader

from dataloader.cifar_dataloader import CustomCIFAR10
from query_optimizer.qo import QueryOptimizer
from pp_models.kde_classifier import KDEClassifier
from config import KDE_MODEL_PATH

class QueryOptimizerCIFAR(QueryOptimizer):
    def __init__(self, query, ml_udf):
        super().__init__(query, ml_udf)

    def setup_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CustomCIFAR10(root='data', train=False, transform=transform)
        return DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # KDE perform best for CIFAR-10 dataset
    def load_pp(self):
        # Load the pre-trained KDE model
        pp_model = KDEClassifier.load(f'{KDE_MODEL_PATH}/{self.query}.pkl')
        return pp_model

    def execute_pp(self, inputs):
        return self.pp.predict(inputs)