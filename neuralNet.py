from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.modules import SoftmaxLayer, TanhLayer, BiasUnit
from neurons import reluLayer
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.structure import FullConnection
from pybrain.tests.helpers import gradientCheck

import neurons


def createNet():
    net = FeedForwardNetwork()
    modules = add_modules(net)
    add_connections(net, modules)
    # finish up
    net.sortModules()
    gradientCheck(net)
    return net


def add_modules(net):
    modules = {}

    #define modules
    modules['inp'] = LinearLayer(40)
    modules['h1'] = reluLayer(20)
    modules['outp'] = SoftmaxLayer(2)

    # add modules
    net.addInputModule(modules['inp'])
    net.addOutputModule(modules['outp'])
    net.addModule(modules['h1'])

    return modules


def add_connections(net, modules):

    net.addConnection(FullConnection(modules['inp'],modules['h1']))
    net.addConnection(FullConnection(modules['h1'], modules['outp']))
