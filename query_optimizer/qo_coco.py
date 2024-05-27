import torch
from torch.utils.data import DataLoader

from query_optimizer.qo import QueryOptimizer
from dataloader.coco_dataloader import COCODataset
from pp_models.dnn_classifier import DNN
from config import COCO_IMAGE_DIR, COCO_ANNOTATION_DIR, DNN_MODEL_PATH


class QueryOptimizerCOCO(QueryOptimizer):
    def __init__(self, query, ml_udf):
        super().__init__(query, ml_udf)

    def setup_dataloader(self):
        dataset = COCODataset(COCO_IMAGE_DIR, COCO_ANNOTATION_DIR)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
        return data_loader
    
    # DNN perform best for COCO dataset
    def load_pp(self):
        pp_model = DNN()
        pp_model.load_state_dict(torch.load(f'{DNN_MODEL_PATH}/{self.query}.pth'))
        return pp_model
    
    def execute_pp(self, inputs):
        return self.pp(inputs)