from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
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
    #gradientCheck(net)
    return net


def add_modules(net):
    modules = {}

    #define modules
    modules['inp'] = SigmoidLayer(800)
    modules['h1'] = SigmoidLayer(1000)
    modules['h3'] = neurons.cosineSimiliarity(400)
    modules['h3'] = neurons.euclideanDistance(800)
    modules['outp'] = LinearLayer(1)

    # add modules
    net.addInputModule(modules['inp'])
    net.addOutputModule(modules['outp'])
    net.addModule(modules['h1'])
    net.addModule(modules['h2'])
    net.addModule(modules['h3'])

    return modules


def add_connections(net, modules):
    # create connections
    m = MotherConnection(400000)

    c1 = SharedFullConnection(m, modules['inp'], modules['h1'], inSliceTo=400)
    c2 = SharedFullConnection(m, modules['inp'], modules['h1'], inSliceFrom=400)
    net.addConnection(c1)
    net.addConnection(c2)
    net.addConnection(FullConnection(modules['h1'], modules['h2']))
    net.addConnection(FullConnection(modules['h2'], modules['h3']))
    net.addConnection(FullConnection(modules['h3'], modules['h4']))
