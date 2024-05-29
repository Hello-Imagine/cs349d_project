import torch

from query_optimizer.qo import QueryOptimizer
from dataloader.imagenet_dataloader import get_imagenet_val_loader
from pp_models.dnn_classifier import DNN
from config import IMAGENET_VAL_DIR, DNN_MODEL_PATH
from query_optimizer.utils import convert_to_int

class QueryOptimizerImageNet(QueryOptimizer):
    def __init__(self, query, ml_udf):
        super().__init__(query, ml_udf)

    def setup_dataloader(self):
        dataloader = get_imagenet_val_loader(IMAGENET_VAL_DIR)
        return dataloader
    
    # DNN perform best for ImageNet dataset
    def load_pp(self, accuracy):
        pp_model = DNN(DNN_MODEL_PATH)
        pp_model.load_state_dict(torch.load(f'{DNN_MODEL_PATH}/{convert_to_int(accuracy)}_{self.query}.pth'))
        return pp_model
    
    def execute_pp(self, inputs):
        return self.pp(inputs)