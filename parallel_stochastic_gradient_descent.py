import numpy as np
import math


# transforms X into m-dimensional feature vectors using RFF and RBF kernel
def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # m is the dimension of the transformed feature vector
    m = 1000

    if X.ndim == 1:
        w = np.random.randn(m)
        return math.sqrt(2) * np.cos(np.inner(w, x) + b)
    else:
        ret = np.zeros([X.shape[0], m])
        for i in range(X.shape[0]):
            w = np.random.randn(m)
            b = np.random.rand() * 2 * np.pi
            ret[i, :] = math.sqrt(2) * np.cos(np.inner(w, X[i]) + b)
        return ret


def mapper(key, value):
    # key: None
    # value: one line of input file

    # 2D NumPy array containing the original feature vectors
    features = np.zeros([5000,400])# this [5000,401] could be a more flexible coding
    # 1D NumPy array containing the classifications of the training data
    classifications = np.zeros(5000)

    # populate features and classifications
    for i in range(len(list)):
        tokens = list[i].split()
        classifications[i] = tokens[0]
        for j in range(1, len(tokens)):
            features[i, j] = float(tokens[j])

    # project features into higher dimensional space
    features = transform(features)

    iterations = 50
    w = np.zeros(1000)
    t = 1
    for i in range(iterations):
        w = update_weights(w, features, classifications, t)
        t += 1

    yield "key", w  # This is how you yield a key, value pair


def update_weights(w, features, classifications, t):
    LAMBDA = 0.001
    eta = 1 / (t * LAMBDA)

    # let's find delta t
    summed_loss = 0
    for i in range(features.shape[0]):
        summed_loss += hinge_loss(w, features[i], classifications[i])
    delta_t = LAMBDA * w - (eta / 5000) * summed_loss

    w_prime = w - eta * delta_t

    return min(1, (1 / math.sqrt(LAMBDA) / w_prime.size)) * w_prime


def hinge_loss(w, x, y):
    max(0, 1 - y * np.inner(w, x))


def reducer(key, values):
    average = np.zeros(1000)

    for w in values:
        average += w

    yield average / (80000 / 5000)
