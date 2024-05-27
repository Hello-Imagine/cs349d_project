from torchvision.transforms import Compose, Resize, ToTensor

from dataloader.voc_dataloader import VOCDataLoader
from query_optimizer.qo import QueryOptimizer
from pp_models.kde_classifier import KDEClassifier
from config import VOC_IMAGE_DIR, VOC_ANNOTATION_DIR, KDE_MODEL_PATH

class QueryOptimizerVOC(QueryOptimizer):
    def __init__(self, query, ml_udf):
        super().__init__(query, ml_udf)
    
    def setup_dataloader(self):
        transform = Compose([
            Resize((640, 640)),
            ToTensor()
        ])
        return VOCDataLoader(VOC_IMAGE_DIR, VOC_ANNOTATION_DIR, transform=transform).get_data_loader()
    
    # KDE perform best for VOC dataset
    def load_pp(self):
        # Load the pre-trained KDE model
        pp_model = KDEClassifier.load(f'{KDE_MODEL_PATH}/{self.query}.pkl')
        return pp_model
    
    def execute_pp(self, inputs):
        return self.pp.predict(inputs)