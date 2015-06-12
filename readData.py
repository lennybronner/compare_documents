from pybrain.datasets.classification import ClassificationDataSet
from pybrain.datasets.importance import ImportanceDataSet
import json
import numpy as np
import itertools
import random
from ExtendedClassificationDataSet import *


def get_data():
    train_ds = ExtendedClassificationDataSet(400, 1, nb_classes=2)
    test_ds = ExtendedClassificationDataSet(400, 1, nb_classes=2)
    print "reading in data..."
    datafile = 'full_text_800.json'
    test_filename = 'test_ds'
    train_filename = 'train_ds'
    random.seed(10)
    train_file = open(train_filename, 'w')
    test_file = open(test_filename, 'w')
    num_zero = 0
    num_one = 0
    with open(datafile) as f:
        for ten_lines in itertools.izip_longest(*[f]*10):
            rand = random.random()
            for line in ten_lines:
                datum = json.loads(line)
                v1 = datum['v1']
                v2 = datum['v2']
                try:
                    v = []
                    for i in range(len(v1)):
                        v.append(v1[i]-v2[i])
                except:
                    continue
                if len(v) == 400:
                    inp = v
                    target = [datum['target']]
                    if rand < 0.8:
                        if target[0] == 0:
                            train_ds.addSample(inp, np.array(target), np.array([1]))
                        else:
                            train_ds.addSample(inp, np.array(target), np.array([3]))
                        train_file.write(str(v1))
                        train_file.write("\n")
                    else:
                        if target[0] == 0:
                            test_ds.addSample(inp, np.array(target), np.array([1]))
                        else:
                            test_ds.addSample(inp, np.array(target), np.array([3]))
                        test_file.write(str(v1))
                        train_file.write("\n")
    print "num of zeros ", num_zero
    print "num of ones ", num_one
    train_file.close()
    test_file.close()
    return train_ds, test_ds
