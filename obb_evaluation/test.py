import numpy as np
import supervision as sv

detections = '''1140.2735239645037 312.66920037162254 1221.6979483633554 256.35700709773494 1353.661876035496 447.1697996283774 1272.2374516366444 503.481992902265 1 0
730.5219934437529 558.6809386910761 1038.3427579323638 421.5117841786743 1095.734006556247 550.303261308924 787.9132420676361 687.4724158213257 2 0
-77.60451149352431 490.9227470887283 39.78217658200708 293.1340581738717 285.72811149352435 439.1018529112717 168.34142341799293 636.8905418261282 0 0
351.36016508257325 200.37968981430797 694.5932261423554 51.8298133568303 887.2314349174267 496.9315101856921 543.9983738576445 645.4813866431698 0 0
1060.8321102693897 90.57715649935015 1278.8040626054137 87.08030150921788 1281.5790897306103 260.0580435006499 1063.6071373945863 263.5548984907822 2 0
-11.341788142963267 159.6878639128999 -0.05617909839081392 55.29612559758558 164.98218814296325 73.13813608710012 153.69657909839083 177.52987440241444 2 0'''

targets = '''1140.2735239645037 312.66920037162254 1221.6979483633554 256.35700709773494 1353.661876035496 447.1697996283774 1272.2374516366444 503.481992902265 1 0
730.5219934437529 558.6809386910761 1038.3427579323638 421.5117841786743 1095.734006556247 550.303261308924 787.9132420676361 687.4724158213257 2 0
-77.60451149352431 490.9227470887283 39.78217658200708 293.1340581738717 285.72811149352435 439.1018529112717 168.34142341799293 636.8905418261282 0 0
351.36016508257325 200.37968981430797 694.5932261423554 51.8298133568303 887.2314349174267 496.9315101856921 543.9983738576445 645.4813866431698 0 0
1060.8321102693897 90.57715649935015 1278.8040626054137 87.08030150921788 1281.5790897306103 260.0580435006499 1063.6071373945863 263.5548984907822 2 0
-11.341788142963267 159.6878639128999 -0.05617909839081392 55.29612559758558 164.98218814296325 73.13813608710012 153.69657909839083 177.52987440241444 2 0'''


def text2nparray(txt: str) -> np.ndarray:
    '''
    Convert text to numpy array
    :param txt: text
    :return: numpy array
    '''
    lines = txt.split('\n')
    result = np.array([list(map(float, line.split(' '))) for line in lines])
    return result


detections = text2nparray(detections)
targets = text2nparray(targets)

class_name = ['paper', 'metal', 'plastic', 'nilon', 'glass', 'fabric']

sv.MeanAveragePrecision.from_detections(detections, targets)
sv.ConfusionMatrix.from_detections(detections, targets, class_name)
