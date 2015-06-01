from pybrain.datasets.classification import ClassificationDataSet
import json
import numpy as np


def get_data():
    ds = ClassificationDataSet(800, 1, nb_classes=2)
    print "reading in data..."
    datafile = 'vectors.txt'
    with open(datafile) as f:
        for line in f:
            datum = json.loads(line)
            v1 = datum['v1']
            v2 = datum['v2']
            v1.extend(v2)
            if len(v1) == 800:
                inp = v1
                target = [datum['target']]
                ds.addSample(inp, np.array(target))
    return ds
