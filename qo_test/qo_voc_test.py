from query_optimizer.qo_voc import QueryOptimizerVOC
from ml_udf.yolov5.yolo_detector import YOLOv5SegmentationDetector
from config import QUERY_ITEM

# Use Yolov5 as the ML UDF
detector = YOLOv5SegmentationDetector('yolov5s.pt', conf_thresh=0.1)

# Create a QueryOptimizerCIFAR object
qo_cifar = QueryOptimizerVOC(QUERY_ITEM, detector)
detected, selectivity, data_reduction = qo_cifar.run()

# Print the detected objects and selectivity
print(f"Detected objects names: {detected}")
print(f"Selectivity: {selectivity}")
print(f"Data reduction: {data_reduction}")