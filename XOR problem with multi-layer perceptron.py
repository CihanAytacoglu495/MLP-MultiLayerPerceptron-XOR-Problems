from model import Model
import numpy as np

inputs = [[0,0], [0,1], [1,0], [1,1]]
outputs = [0, 1, 1, 0]

m = Model()

m.train(inputs, outputs)

for i in inputs:
    p = m.predict(i)
    print (str(i) + ' => ' + str(p))
    from neuron import HiddenNeuron, OutputNeuron
import numpy as np

    class Model(object):

    def __init__(self):
        self.hidden = [HiddenNeuron(2) for i in range(2)]
        self.output = OutputNeuron(2)

    def predict(self, input):
        temp = []
        for x in range(2):
            self.hidden[x].forward(input)
            temp.append(self.hidden[x].out)
        self.output.forward(temp)
        return self.output.out

    def train(self, inputs, targets):
        it = 0
        i = 0
        size = len(inputs)
        while it < 4:
            if i == size:
                i = 0
            feature = inputs[i]
            print '\n\nFeature : ' + str(feature) + '\n'
            print 'Output weights : ' + str(self.output.weights)
            print 'Hidden 1 weights : ' + str(self.hidden[0].weights)
            print 'Hidden 2 weights : ' + str(self.hidden[1].weights)
            temp = []
            for x in range(2):
                self.hidden[x].forward(feature)
                temp.append(self.hidden[x].out)
            self.output.forward(temp)
            self.output.backward(targets[i])
            deltas = []
            deltas.append(self.output.error)
            weights = []
            weights.append([self.output.weights[0]])
            weights.append([self.output.weights[1]])
            for x in range(2):
                self.hidden[x].backward(deltas, weights[x])
            for x in range(2):
                self.hidden[x].update(feature)
            self.output.update(temp)
            it += 1
            i += 1
            import numpy as np
from random import uniform

class Neuron(object):

    def activation(self, fx):
        return 1/(1 + np.exp(-fx))

    def __init__(self, dim, lrate):
        self.dim = dim
        self.weights = np.empty([dim])
        self.weights = [uniform(0,1) for x in range(dim)]
        self.bias = uniform(0, 1)
        self.lrate = lrate
        self.out = None
        self.error = None

    def update(self, input):
        j = 0
        for i in input:
            delta = self.lrate * self.error
            self.weights[j] -= (delta*i)
            self.bias += delta
            j+=1

    def forward(self, input):
        j = 0
        sum = self.bias
        for f in input:
            sum += f * self.weights[j]
            j+=1
        self.out = self.activation(sum)

    def backward(self):
        pass

class OutputNeuron(Neuron):

    def __init__(self, dim, lrate=0.2):
        super(OutputNeuron, self).__init__(dim, lrate)

    def backward(self, target):
        self.error = self.out * (1 - self.out) * (self.out - target)


class HiddenNeuron(Neuron):

    def __init__(self, dim, lrate=0.2):
        super(HiddenNeuron, self).__init__(dim, lrate)

    def backward(self, deltas, weights):
        sum = 0
        size = len(deltas)
        for x in range(size):
            sum += deltas[x] * weights[x]
        self.error = self.out * (1 - self.out) * sum