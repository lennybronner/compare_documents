from pybrain.supervised.trainers import BackpropTrainer

import conf
import neuralNet
import readData
import numpy as np


def runNet():
    net = neuralNet.createNet()
    ds = readData.get_data()
    print len(ds)
    train_model(net, ds)


def train_model(net, ds):
    # train model
    train, test = ds.splitWithProportion(0.75)
    trainer = BackpropTrainer(net, train, learningrate=0.0001)

    #print trainer.trainUntilConvergence(maxEpochs=100)

    counter = []
    for i in range(10):
        guess = []
        correct = []
        print trainer.train()
        count = 0
        for i in range(len(test)):
            output = net.activate(test['input'][i])[0]
            output = int(round(output))
            real = int(test['target'][i][0])
            guess.append(output)
            correct.append(real)
            if output == real:
                count += 1
        print count
        counter.append(count)
        conf_arr = np.zeros((2, 2))
        for j in range(len(guess)):
            conf_arr[guess[j]][correct[j]] += 1
        conf.makeconf(conf_arr)
    print counter
