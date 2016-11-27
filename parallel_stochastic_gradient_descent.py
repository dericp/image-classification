import numpy as np

# constants
# m is the dimension of the transformed feature vector
m = 3000
# number of iterations of PEGASOS
iterations = 300000
# regularization constant
lambda_val = 1e-6
# standard deviation of p in RFF
sigma = 10


# transforms X into m-dimensional feature vectors using RFF and RBF kernel
# Make sure this function works for both 1D and 2D NumPy arrays.
def transform(X):
    np.random.seed(0)
    b = np.random.rand(m) * 2 * np.pi

    if X.ndim == 1:
        w = np.random.multivariate_normal(np.zeros(X.size), sigma**2 * np.identity(X.size), m)
    else:
        w = np.random.multivariate_normal(np.zeros(X.shape[1]), sigma**2 * np.identity(X.shape[1]), m)

    transformed = (2.0 / m)**0.5 * np.cos(np.dot(X, np.transpose(w)) + b)
    # feature normalization
    transformed = (transformed - np.mean(transformed, 0)) / np.std(transformed, 0)

    return transformed


# key: None
# value: one line of input file
def mapper(key, value):
    # 2D NumPy array containing the original feature vectors
    features = np.zeros([len(value), len(value[0].split()) - 1])
    # 1D NumPy array containing the classifications of the training data
    classifications = np.zeros(len(value))

    # populate features and classifications
    for i in range(len(value)):
        tokens = value[i].split()
        classifications[i] = tokens[0]
        features[i] = tokens[1:]

    # project features into higher dimensional space
    features = transform(features)

    # PEGASOS
    w = np.zeros(m)
    for i in range(1, iterations):
        w = update_weights(w, features, classifications, i)

    yield 0, w


# weight vector update of PEGASOS
def update_weights(w, features, classifications, t):
    i = int(np.random.uniform(0, features.shape[0]))
    learning_rate = 1 / (lambda_val * t)

    new_w = (1 - learning_rate * lambda_val) * w + learning_rate * hinge_loss_gradient(w, features[i], classifications[i])
    # optional projection step
    #new_w = min(1, ((1 / lambda_val**0.5) / np.linalg.norm(new_w))) * new_w

    return new_w


# calculate the gradient of the hinge loss function
def hinge_loss_gradient(w, x, y):
    if np.dot(w, x) * y >= 1:
        return 0
    else:
        return y * x


def reducer(key, values):
    cumulative_weights = np.zeros(m)

    for w in values:
        cumulative_weights += w

    # yield the average of the weights
    yield cumulative_weights / len(values)
