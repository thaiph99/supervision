import numpy as np
import supervision as sv
import os

anno_path = '/home/thaiph/data/data_waste_original/train/labelTxt'
with open(os.path.join(anno_path, 'demo3_frame50.txt'), 'r') as f:
    detections = f.readlines()
targets = detections


def text2nparray(txts: str) -> np.ndarray:
    """
    Convert text to numpy array
    :param txt: text
    :return: numpy array
    """
    lines = [txt.strip() for txt in txts]
    result = np.array([list(map(float, line.split(' '))) for line in lines])
    return result


detections = text2nparray(detections)
targets = text2nparray(targets)

class_name = ['paper', 'metal', 'plastic', 'nilon', 'glass', 'fabric', 'other']
print("class_name")
print(class_name)
print(detections)
print(targets)
print('check shape: ', detections.shape)


# sv.MeanAveragePrecision.from_detections(detections, targets)
# sv.ConfusionMatrix.from_detections(detections, targets, class_name)
