import json
import random

class Network:
    def __init__(self, layers, learning_rate, activation):
        self.layers = layers
        self.weights = [Matrix.random(layers[i + 1], layers[i]) for i in range(len(layers) - 1)]
        self.biases = [Matrix.random(layers[i + 1], 1) for i in range(len(layers) - 1)]
        self.data = []
        self.learning_rate = learning_rate
        self.activation = activation

    def feed_forward(self, inputs):
        if len(inputs) != self.layers[0]:
            raise ValueError("Invalid inputs length")
        
        current = Matrix.from_list([inputs]).transpose()
        self.data = [current]

        for i in range(len(self.layers) - 1):
            current = self.weights[i].multiply(current).add(self.biases[i]).map(self.activation.function)
            self.data.append(current)

        return current.transpose().data[0]

    def back_propagate(self, outputs, targets):
        if len(targets) != self.layers[-1]:
            raise ValueError("Invalid targets length")

        parsed = Matrix.from_list([outputs]).transpose()
        errors = Matrix.from_list([targets]).transpose().subtract(parsed)
        gradients = parsed.map(self.activation.derivative)

        for i in range(len(self.layers) - 2, -1, -1):
            gradients = gradients.dot_multiply(errors).map(lambda x: x * self.learning_rate)

            self.weights[i] = self.weights[i].add(gradients.multiply(self.data[i].transpose()))
            self.biases[i] = self.biases[i].add(gradients)

            errors = self.weights[i].transpose().multiply(errors)
            gradients = self.data[i].map(self.activation.derivative)

    def train(self, inputs, targets, epochs):
        for i in range(1, epochs + 1):
            ''' 
            if epochs < 100 or i % (epochs // 100) == 0:
                print(f"Epoch {i} of {epochs}")
            '''
            for j in range(len(inputs)):
                outputs = self.feed_forward(inputs[j])
                self.back_propagate(outputs, targets[j])

    def save(self, file):
        save_data = {
            "weights": [matrix.data for matrix in self.weights],
            "biases": [matrix.data for matrix in self.biases],
        }

        with open(file, "w") as f:
            json.dump(save_data, f)

    def load(self, file):
        with open(file, "r") as f:
            save_data = json.load(f)

        weights = [Matrix.from_list(data) for data in save_data["weights"]]
        biases = [Matrix.from_list(data) for data in save_data["biases"]]

        self.weights = weights
        self.biases = biases

class Matrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]

    @classmethod
    def zeroes(cls, rows, cols):
        return cls(rows, cols)

    @classmethod
    def random(cls, rows, cols):
        matrix = cls.zeroes(rows, cols)
        for i in range(rows):
            for j in range(cols):
                matrix.data[i][j] = random.uniform(-1.0, 1.0)
        return matrix

    @classmethod
    def from_list(cls, data):
        rows = len(data)
        cols = len(data[0])
        matrix = cls(rows, cols)
        matrix.data = data
        return matrix

    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("Attempted to multiply two matrices with incorrect dimensions")
        res = Matrix.zeroes(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                sum = 0.0
                for k in range(self.cols):
                    sum += self.data[i][k] * other.data[k][j]
                res.data[i][j] = sum
        return res

    def add(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Attempted to add matrices with incorrect dimensions")
        res = Matrix.zeroes(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                res.data[i][j] = self.data[i][j] + other.data[i][j]
        return res

    def dot_multiply(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Attempted to dot multiply matrices with incorrect dimensions")
        res = Matrix.zeroes(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                res.data[i][j] = self.data[i][j] * other.data[i][j]
        return res

    def subtract(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Attempted to subtract matrices with incorrect dimensions")
        res = Matrix.zeroes(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                res.data[i][j] = self.data[i][j] - other.data[i][j]
        return res

    def map(self, function):
        return Matrix.from_list([[function(value) for value in row] for row in self.data])

    def transpose(self):
        res = Matrix.zeroes(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                res.data[j][i] = self.data[i][j]
        return res

