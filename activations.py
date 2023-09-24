import math

class Activation:
    def __init__(self,function,derivative):
        self.function = function
        self.derivative = derivative


IDENTITY = Activation(lambda x: x, 
         1.0)

SIGMOID = Activation(lambda x: 1.0 / (1.0 + math.e ** (-x)), 
        lambda x: x * (1.0 - x))

TANH = Activation(lambda x: math.tanh(x), 
        lambda x: 1.0 - x**2)

RELU = Activation(lambda x: max(x,0.0),
        lambda x: 1.0 if x > 0.0 else 0.0 )

LEAKY_RELU = Activation(lambda x: x if x>0.0 else 0.01*x,
        lambda x: 1.0 if x > 0.0 else 0.01 )
