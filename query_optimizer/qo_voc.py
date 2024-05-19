import torch
from torchvision.transforms import Compose, Resize, ToTensor

from dataloader.voc_dataloader import VOCDataLoader
from query_optimizer.qo import QueryOptimizer

VOC_IMAGE_DIR = 'data/VOC2012/JPEGImages'
VOC_ANNOTATION_DIR = 'data/VOC2012/Annotations'

class QueryOptimizerVOC(QueryOptimizer):
    def setup_dataloader(self):
        transform = Compose([
            Resize((640, 640)),
            ToTensor()
        ])
        return VOCDataLoader(VOC_IMAGE_DIR, VOC_ANNOTATION_DIR, transform=transform).get_data_loader()
    
    def load_pp_model(self):
        pp_model = DNN()
        pp_model.load_state_dict(torch.load('models/voc.pt'))
        pp_model.eval()
        return pp_model