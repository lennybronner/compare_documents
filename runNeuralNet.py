from pybrain.supervised.trainers import BackpropTrainer
from ExtendedBackprop import ExtendedBackpropTrainer

import conf
#import neuralNet2 as neuralNet
import neuralNet
import readData
import numpy as np
import scipy as sp
import random

from pybrain.utilities import percentError

import pylab as pl
from sklearn import svm, datasets
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

from scipy import diag, arange, meshgrid, where, random
from numpy.random import multivariate_normal, seed

from pybrain.datasets.classification import ClassificationDataSet

def runNet():
    np.random.seed(10)
    sp.random.seed(10)
    net = neuralNet.createNet()
    train_ds, test_ds = readData.get_data()
    train_model(net, train_ds, test_ds)


def train_model(net, train_ds, test_ds):

    # train model
    tstdata, trndata = test_ds, train_ds
    tstdata._convertToOneOfMany()
    trndata._convertToOneOfMany()

    print "Number of training patterns: ", len(trndata)
    print "Number of test patterns: ", len(tstdata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim

    trainer = ExtendedBackpropTrainer(net, learningrate=0.01, dataset=trndata, verbose=True, lrdecay=.99999, weightdecay=0.001)

    for i in range(20):
        trainer.trainEpochs(5)
        trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
        tstresult = percentError( trainer.testOnClassData(
               dataset=tstdata ), tstdata['class'] )

        # Compute ROC curve and area the curve
        probas_ = net.activateOnDataset(tstdata)
        fpr, tpr, thresholds = roc_curve(tstdata['class'], probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        print "Area under the ROC curve : %f" % roc_auc

        print "epoch: %4d" % trainer.totalepochs, \
              "  train error: %5.2f%%" % trnresult, \
              "  test error: %5.2f%%" % tstresult

        guess = []
        correct = []
        count = 0
        for i in range(len(tstdata)):
            output = net.activate(tstdata['input'][i])[0]
            output = int(round(output))
            real = int(tstdata['target'][i][0])
            guess.append(output)
            correct.append(real)
            if output == real:
                count += 1
        conf_arr = np.zeros((2, 2))
        for j in range(len(guess)):
            conf_arr[guess[j]][correct[j]] += 1
        print conf_arr
    
    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()

if __name__ == '__main__':
    runNet()
