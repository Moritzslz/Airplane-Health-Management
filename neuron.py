import numpy as np


def sum(input_vector, weight_vector, bias):
    """
    This method operates according to the math of a feed-forward neural network

    inputs i, weights w, bias b, -> output
    f(i1 * w1 + i2 * w2 + ... + in * wn + b) = output

    :param:
    input_vector ([float]): both vectors need to have the same length
    weight_vector ([float]): both vectors need to have the same length

    :return:
    output == 1 => meaning engine is in good shape
    output == 0 => meaning engine is in bad shape
    """

    if len(input_vector) != len(weight_vector):
        print("Input vector length != weight vector length")
        return

    weighted_sum = np.dot(input_vector, weight_vector) + bias
    return activation_function(weighted_sum)


def activation_function(x):
    """
    The chosen activation function is the step function
    :param:
    x (float)
    """
    return 1 if x > 0 else 0



if __name__ == '__main__':
    weight_vector = np.array([2, -0.5, -0.5, 1])  # Example weight vector
    # Good
    input_vector_good = np.array([320, 3100, 111.8, 43.2])  # Example input features for an observation
    # Bad
    input_vector_bad = np.array([320, 500, 70, 95.0])  # Example input features for an observation
    bias = 0.1
    output_1 = sum(input_vector_good, weight_vector, bias)
    output_2 = sum(input_vector_bad, weight_vector, bias)
    print("Output 1:", output_1)
    print("Output 2:", output_2)
