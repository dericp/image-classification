import numpy as np
import math

m = 400
iterations = 500
lambda_val = 0.001


# transforms X into m-dimensional feature vectors using RFF and RBF kernel
def transform(X):
    print('in transform method')
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # m is the dimension of the transformed feature vector

    if X.ndim == 1:
        print('X was one dimensional ' + str(X.shape))
        ret = np.zeros(m)
        w = np.random.randn(m, X.size)
        b = np.random.rand(m) * 2 * np.pi

        for i in range(m):
            ret[i] = math.sqrt(2) * np.cos(np.dot(w[i], X) + b[i])

        return ret
    else:
        print('X was not one dimensional ' + str(X.shape))
        ret = np.zeros([X.shape[0], m])
        w = np.random.randn(m, X.shape[1])
        b = np.random.rand(m) * 2 * np.pi

        for i in range(X.shape[0]):
            print('transforming the ' + str(i) + ' feature vector')
            for j in range(m):
                ret[i][j] = np.cos(np.dot(w[j], X[i]) + b[j])

        return math.sqrt(2) * ret


def mapper(key, value):
    print('got into the mapper')
    # key: None
    # value: one line of input file

    # 2D NumPy array containing the original feature vectors
    features = np.zeros([5000,400])# this [5000,401] could be a more flexible coding
    print('made empty features')
    # 1D NumPy array containing the classifications of the training data
    classifications = np.zeros(5000)
    print('made empty classifications')

    # populate features and classifications
    for i in range(len(value)):
        tokens = value[i].split()
        classifications[i] = tokens[0]
        for j in range(0, len(tokens) - 1):
            features[i, j] = float(tokens[j])

    print('populated features and classifications')

    # project features into higher dimensional space
    features = transform(features)
    print('transformed features. features now dim ' + str(features.shape))

    w = np.zeros(m)
    print('starting gradient descent')
    for i in range(1, iterations):
        print('on timestep ' + str(i))
        w = update_weights(w, features, classifications, i)

    yield 'key', w  # This is how you yield a key, value pair


def update_weights(w, features, classifications, t):
    # let's find delta t

    eta = 1 / (t * lambda_val)
    summed_loss = 0
    for i in range(features.shape[0]):
        summed_loss += hinge_loss(w, features[i], classifications[i])
    delta_t = lambda_val * w - (eta / 5000) * summed_loss

    w_prime = w - eta * delta_t

    return min(1, (1 / math.sqrt(lambda_val) / w_prime.size)) * w_prime


def hinge_loss(w, x, y):
    return max(0, 1 - y * np.dot(w, x))


def reducer(key, values):
    average = np.zeros(m)

    for w in values:
        #feature_vector = np.fromstring(w)
        average += w

    yield average / (80000 / 5000)
