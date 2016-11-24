import numpy as np
from numpy import linalg as LA

def list2array(list):
    input_2darray = np.zeros([5000,401])# this [5000,401] could be a more flexible coding
    for i in range(len(list)):
        token = list[i].split(' ')
        for j in range(len(token)):
            input_2darray[i,j] = float(token[j])
    return input_2darray

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return X

def computeFF(x,y,m): # this code is for RBF kernel
    # the input the x,y are d-dim; want to map them into D dim; then "inverse" them to m dim
    # output: the expectation of zx*zy, i.e. the approximate k(x-y)
    #rand_feat_w = np.zeros([m,D])# initialize the feature matrice
    #rand_feat_b = np.zeros([m,1])
    zxzy = 0
    for i in range(m):
        w = np.random.normal(0,1,[D,1])
        b = np.pi*np.random.rand(1)
        zx = np.sqrt(2)*np.cos(np.inner(w,x)+b)
        zy = np.sqrt(2)*np.cos(np.inner(w,y)+b)
        #rand_feat_w[i,:] = np.transpose(w)
        #rand_feat_b[i,:] = np.transpose(b)
        zxzy += zx*zy/m
    return zxzy

# def extract_nystrom_feat(X,m):
#     # input: X is n*d np.array contains n features: x1,...,xn; m is the size of sampling
#     n = X.shape[0]
#     rand_idx = np.random.choice(m,m,replace=False)
#     S = X[rand_idx,:]
#     KSS = np.zeros([m,m])
#     D1 = 10*m # hard code
#     m1 = 2*m # hard code
#     for i in range(m):
#         KSS[i,i] = computeFF(S[i,:],S[i,:],m1,D1)
#         for j in range(i+1,m,1):
#             KSS[i,j] = computeFF(S[i,:],S[j,:],m1,D)
#             KSS[j,i] = KSS[i,j]
#     eigvalue,eigvectors = LA.eig(KSS)
#     D = np.diag(eigvalue)
#     Khat = np.zeros([n,n])
#     for i in range(n):
#         KSi = np,zeros([m,1])
#         for k in range(m):
#             KSi[j] = computeFF(X[i,:],S[k,:],m1,D1)
#         zxi = np.matmul(1./np.sqrt(D),KSi)
#         Khat[i,i] = np.inner(zxi,zxi)
#         for j in range(n):
#             # maybe there's a faster way than compute the k(xi,sj) again ina for loop
#             for k in range(m):
#                 KSj = computeFF(X[i,:],S[k,:],m1,D1)
#             zxj = np.matmul(1./np.sqrt(D),KSj)
#             Khat[i,j] = np.inner(zxi,zxj)
#             Khat[j,i] = np.inner(zxi,zxj)
#     return Khat

def mapper(key, value):
    # key: None
    # value: one line of input file
    value = list2array(value)
    for line in value:
     #   tokens = line.split()
        y = line[0]
        features = line[1:]

        print len(features)
    yield "key", "value"  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.random.randn(400)
