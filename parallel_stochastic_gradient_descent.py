import numpy as np
import math

m = 1000
iterations = 500000
lambda_val = 1e-8
sigma = 10;


# transforms X into m-dimensional feature vectors using RFF and RBF kernel
def transform(X):
    print('in transform method')
    # Make sure this function works for both 1D and 2D NumPy arrays.
    # m is the dimension of the transformed feature vector

    #w = sigma * np.random.randn(m, X.shape[1])
    #w = sigma * np.random.randn(m, X.size)

    b = np.random.rand(m) * 2 * np.pi

    if X.ndim == 1:
        print('X was one dimensional ' + str(X.shape))
        w = np.random.multivariate_normal(np.zeros(X.size), 100 * np.identity(X.size), m)
        ret = np.zeros(m)

        for i in range(m):
            ret[i] = math.sqrt(2) * np.cos(np.dot(w[i], X) + b[i])

        return ret
    else:
        print('X was not one dimensional ' + str(X.shape))
        w = np.random.multivariate_normal(np.zeros(X.shape[1]), 100 * np.identity(X.shape[1]), m)
        ret = np.zeros([X.shape[0], m])

        for i in range(X.shape[0]):
            #print('transforming the ' + str(i) + ' feature vector')
            for j in range(m):
                ret[i][j] = np.cos(np.dot(w[j], X[i]) + b[j])

        return math.sqrt(2) * ret


def mapper(key, value):
    #print('got into the mapper')
    # key: None
    # value: one line of input file

    # 2D NumPy array containing the original feature vectors
    features = np.zeros([5000,400])# this [5000,401] could be a more flexible coding
    #print('made empty features')
    # 1D NumPy array containing the classifications of the training data
    classifications = np.zeros(5000)
    #print('made empty classifications')

    # populate features and classifications
    for i in range(len(value)):
        tokens = value[i].split()
        classifications[i] = tokens[0]
        for j in range(1, len(tokens)):
            features[i, j - 1] = float(tokens[j])

    #print('populated features and classifications')

    # project features into higher dimensional space
    features = transform(features)
    print('transformed features. features now dim ' + str(features.shape))

    w = np.zeros(m)
    #print('starting gradient descent')
    for i in range(1, iterations):
        #print('on timestep ' + str(i))
        w = update_weights(w, features, classifications, i)

    yield 'key', w  # This is how you yield a key, value pair


def update_weights(w, features, classifications, t):

    i = int(np.random.uniform(0, features.shape[0]))

    learning_rate = 1 / (lambda_val * t)
    new_w = (1 - learning_rate * lambda_val) * w + learning_rate * hinge_loss_gradient(w, features[i], classifications[i])
    # optional projection step
    new_w = min(1, ((1 / math.sqrt(lambda_val)) / np.linalg.norm(new_w))) * new_w
    return new_w


def hinge_loss_gradient(w, x, y):
    if np.dot(w, x) * y >= 1:
        return 0
    else:
        return y * x


def reducer(key, values):
    cumulative_weights = np.zeros(m)

    for w in values:
        cumulative_weights += w

    # yield the average
    yield cumulative_weights / len(values)
