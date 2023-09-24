from network import Network
from activations import TANH, SIGMOID, RELU, LEAKY_RELU

if __name__ == "__main__":
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    targets = [
        [0.0],
        [1.0],
        [1.0],
        [0.0],
    ]

    print("TANH")
    network = Network([2, 3, 1], 0.5, TANH)
    network.train(inputs.copy(), targets.copy(), 1000)
    print(network.feed_forward([0.0, 0.0]))
    print(network.feed_forward([0.0, 1.0]))
    print(network.feed_forward([1.0, 0.0]))
    print(network.feed_forward([1.0, 1.0]))

    print("SIGMOID")
    network = Network([2, 3, 1], 0.5, SIGMOID)
    network.train(inputs.copy(), targets.copy(), 1000)
    print(network.feed_forward([0.0, 0.0]))
    print(network.feed_forward([0.0, 1.0]))
    print(network.feed_forward([1.0, 0.0]))
    print(network.feed_forward([1.0, 1.0]))

    print("RELU")
    network = Network([2, 3, 1], 0.5, RELU)
    network.train(inputs.copy(), targets.copy(), 1000)
    print(network.feed_forward([0.0, 0.0]))
    print(network.feed_forward([0.0, 1.0]))
    print(network.feed_forward([1.0, 0.0]))
    print(network.feed_forward([1.0, 1.0]))

    print("LEAKY_RELU")
    network = Network([2, 3, 1], 0.5, LEAKY_RELU)
    network.train(inputs.copy(), targets.copy(), 1000)
    print(network.feed_forward([0.0, 0.0]))
    print(network.feed_forward([0.0, 1.0]))
    print(network.feed_forward([1.0, 0.0]))
    print(network.feed_forward([1.0, 1.0]))

