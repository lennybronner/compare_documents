from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.modules import SoftmaxLayer, TanhLayer
from pybrain.structure.connections.shared import MotherConnection, SharedFullConnection
from pybrain.structure import FullConnection
from pybrain.structure.modules import BiasUnit
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
    modules['inp'] = LinearLayer(400)
    modules["input_bias"] = BiasUnit()
    modules['h1'] = TanhLayer(300)
    modules['h1_bias'] = BiasUnit()
    modules['h2'] = TanhLayer(200)
    modules['h2_bias'] = BiasUnit()
    #modules['h3'] = neurons.euclideanDistance(100)
    modules['outp'] = SoftmaxLayer(2)
    modules['output_bias'] = BiasUnit()

    # add modules
    net.addInputModule(modules['inp'])
    net.addOutputModule(modules['outp'])
    net.addModule(modules['h1'])
    net.addModule(modules['h2'])
    net.addModule(modules['input_bias'])
    net.addModule(modules['h1_bias'])
    net.addModule(modules['h2_bias'])
    net.addModule(modules['output_bias'])
    #net.addModule(modules['h3'])

    return modules


def add_connections(net, modules):
    # create connections
    #m = MotherConnection(160000)

    #net.addConnection(FullConnection(modules['input_bias'], modules['inp']))
    #net.addConnection(FullConnection(modules['h1_bias'], modules['h1']))
    #net.addConnection(FullConnection(modules['h2_bias'], modules['h2']))
    #net.addConnection(FullConnection(modules['output_bias'], modules['outp']))

    #c1 = SharedFullConnection(m, modules['inp'], modules['h1'], inSliceTo=400)
    #c2 = SharedFullConnection(m, modules['inp'], modules['h1'], inSliceFrom=400)
    net.addConnection(FullConnection(modules['inp'],modules['h1']))
    net.addConnection(FullConnection(modules['h1'],modules['h2']))

    #net.addConnection(c1)
    #net.addConnection(c2)
    #net.addConnection(FullConnection(modules['h1'], modules['h2']))
    #net.addConnection(FullConnection(modules['h2'], modules['h3']))
    net.addConnection(FullConnection(modules['h2'], modules['outp']))
