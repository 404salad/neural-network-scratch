import random

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
        res = cls.zeroes(rows, cols)
        for i in range(rows):
            for j in range(cols):
                res.data[i][j] = random.uniform(-1.0, 1.0)
        return res

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

    def __str__(self):
        return "Matrix {{\n{}\n}}".format(
            "\n".join(["  " + " ".join(map(str, row)) for row in self.data])
        )

    def __repr__(self):
        return self.__str__()

