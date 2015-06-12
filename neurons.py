from pybrain.structure.connections.connection import Connection
from pybrain.structure.modules.neuronlayer import NeuronLayer
import numpy as np


class SingleConnection(Connection):
    def all_same(self, items):
        return all(x == items[0] for x in items)

    def __init__(self, *args, **kwargs):
        Connection.__init__(self, *args, **kwargs)

    def _forwardImplementation(self, inbuf, outbuf):
        #assert(self.all_same(inbuf))
        outbuf[:] = inbuf[0]

    def _backwardImplementation(self, outerr, inerr, inbuf):
        inerr[:] = outerr*inbuf


class cosineSimiliarity(NeuronLayer):
    def gradCosine(self, v1, v2):
        numerator = v2 * np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2) - (np.dot(v1, v2)*v1*np.linalg.norm(v2, 2))/float(np.linalg.norm(v1, 2))
        denominator = np.linalg.norm(v1, 2)**2 * np.linalg.norm(v2, 2)**2
        grad = numerator/float(denominator)
        return grad

        first = v2/float(np.linalg.norm(v1)*np.linalg.norm(v2))
        second = (np.dot(v1, v2)*v1)/(np.linalg.norm(v1)**3)
        return first - second

    def cosine(self, v1, v2):
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2)
        return float(dot)/float(norm)

    def _forwardImplementation(self, inbuf, outbuf):
        v1 = inbuf[:len(inbuf)/2]
        v2 = inbuf[len(inbuf)/2:]
        cosine = self.cosine(v1, v2)
        print cosine
        outbuf[:] = cosine

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        v1 = inbuf[:len(inbuf)/2]
        v2 = inbuf[len(inbuf)/2:]
        dv1 = self.gradCosine(v1, v2)
        #sin_v1v2 = math.sqrt(1-self.cosine(v1, v2)**2)
        #dv1 = v2*sin_v1v2
        #dv2 = v1*sin_v1v2
        dv2 = self.gradCosine(v2, v1)
        new_grad = np.zeros_like(inbuf)
        new_grad[:len(outerr)/2] = dv1
        new_grad[len(outerr)/2:] = dv2
        inerr[:] = np.dot(new_grad, outerr)

class reluLayer(NeuronLayer):

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = inbuf * (inbuf > 0)
        print outbuf

    def _backwardImplementation(self, outerr, inerr, inbuf, outbuf):
        inerr[:] = outerr * (inbuf > 0)

class euclideanDistance(NeuronLayer):
    def euclid_dist(self, v1, v2):
        diff = v1 - v2
        dist = np.linalg.norm(diff)
        return dist

    def euclid_dist_grad(self, inbuf):
        v1 = inbuf[:len(inbuf)/2]
        v2 = inbuf[len(inbuf)/2:]
        return 2*(v1 - v2), -2*(v1 - v2)

    def _forwardImplementation(self, inbuf, outbuf):
        v1 = inbuf[:len(inbuf)/2]
        v2 = inbuf[len(inbuf)/2:]
        outbuf[:] = (1/2)*self.euclid_dist(v1, v2)**2

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        dv1, dv2 = self.euclid_dist_grad(inbuf)
        new_grad = np.zeros_like(inbuf)
        new_grad[:len(outerr)/2] = dv1
        new_grad[len(outerr)/2:] = dv2
        inerr[:] = np.dot(new_grad, outerr)
