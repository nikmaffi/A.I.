import math
import numpy
import functools

class Neuron:
    def __init__(self, nSynapses, trainingFrequency, learningRate):
        self.trainingFrequency = trainingFrequency
        self.learningRate = learningRate

        self.nSynapses = nSynapses
        self.weights = numpy.random.random(nSynapses)
        self.bias = numpy.random.random()

    def active(self, inputs):
        inputs = numpy.array(inputs, dtype=float)
        sum = functools.reduce(lambda s, el: s + el[0] * el[1], zip(self.weights, inputs),  0) + self.bias

        return 1 / (1 + math.exp(-sum))

    def __derivative(self, inputs, objective, relativeDerivative):
        sigm = self.active(inputs)

        return 2 * (sigm - objective) * (sigm * (1 - sigm)) * relativeDerivative

    def fit(self, inputs, objectives):
        inputs = numpy.array(inputs, dtype=float)
        objectives = numpy.array(objectives, dtype=float)

        for _ in range(self.trainingFrequency):
            ti = numpy.random.randint(0, len(inputs))

            for i in range(self.nSynapses):
                self.weights[i] = self.weights[i] - self.learningRate * self.__derivative(inputs[ti], objectives[ti], inputs[ti][i])
            self.bias = self.bias - self.learningRate * self.__derivative(inputs[ti], objectives[ti], 1)