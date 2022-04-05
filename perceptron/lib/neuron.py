import numpy
import functools

class Neuron:
    def __init__(self, synapses, learning_rate, active_func):
        self.learning_rate = learning_rate
        self.synapses = synapses
        self.active_func = active_func

        self.weights = numpy.random.random(synapses)
        self.bias = numpy.random.random()

    def __active(self, inputs):
        sum = functools.reduce(lambda s, el: s + el[0] * el[1], zip(self.weights, inputs),  0) + self.bias

        return self.active_func(sum)

    def __derivative(self, inputs, output_expected, relativ_derivative):
        func = self.__active(inputs)

        return 2 * (func - output_expected) * (func * (1 - func)) * relativ_derivative

    def score(self, inputs, expected_outputs):
        outputs = self.predict(inputs)

        corrects = (outputs.round() == expected_outputs).sum()
        totals = len(expected_outputs)

        return corrects / totals

    def predict(self, inputs_set):
        outputs = numpy.zeros(len(inputs_set))

        for i, inputs in enumerate(inputs_set):
            outputs[i] = self.__active(inputs)

        return outputs

    def fit(self, inputs, output_expected, training_frequency):
        for _ in range(training_frequency):
            ti = numpy.random.randint(0, len(inputs))

            for i in range(self.synapses):
                self.weights[i] = self.weights[i] - self.learning_rate * self.__derivative(inputs[ti], output_expected[ti], inputs[ti][i])
            self.bias = self.bias - self.learning_rate * self.__derivative(inputs[ti], output_expected[ti], 1)