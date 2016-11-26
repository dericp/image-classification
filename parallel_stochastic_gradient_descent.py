import numpy as np
from numpy import linalg as LA
import random

def list2array(list):
    input_2darray = np.zeros([5000,401])# this [5000,401] could be a more flexible coding
    for i in range(len(list)):
        token = list[i].split(' ')
        for j in range(len(token)):
            input_2darray[i,j] = float(token[j])
    return input_2darray

def transform(X):
    print "******** transform *********"
    print "dim is " + str(X.ndim)
    # Make sure this function works for both 1D and 2D NumPy arrays.
    if X.ndim == 1:
        X = np.append(X,1)
    else:
        print "number of test is " + str(len(X))
        ones = np.ones(len(X))
        X = np.column_stack((X, ones))
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
    w = np.zeros(401)
    print "*********** mapper ********"
    step = 0
    lambda_ = 0.00001
    T = 2000
    k = 50 # k is the size of the randomly chosen subset

    ############ IMPLEMENTATION OF OPGD ############
    # for line in value:
    #     step += 1
    #     y = line[0]
    #     features = np.append(line[1:], 1)
    #     if (y * np.dot(w, features) < 1):
    #         w = w + y*features/np.sqrt(step)
    #         len_w = np.sqrt(np.dot(w,w)/len(w))
    #         #print "len_w is " + str(len_w)
    #         #print "proj: " + str(1/(np.sqrt(lambda_) * len_w))
    #         w = w * min(1, 1/(np.sqrt(lambda_) * len_w))

    ############ IMPLEMENTATION OF PEGASOS ############
    for t in range(T):
        setIndex = random.sample(range(0, 5000), k)
        g_sum = np.zeros(401)
        for i in setIndex:
            line = value[i]
            y = line[0]
            features = np.append(line[1:], 1)
            if (y * np.dot(w, features) < 1):
                g_sum += y*features
        w = (1-1.0/(1+t))*w + g_sum/(k*(t+1)*lambda_)
        size_w = np.sqrt(np.dot(w,w)/len(w))
        w = w * min(1, 1/(np.sqrt(lambda_) * size_w))


    yield "key", w.tostring()  # This is how you yield a key, value pair


def reducer(key, values):
    print "************* reducer ************"
    a = np.empty((16,401))
    for i in range(len(values)):
        a[i] = np.fromstring(values[i])
    print "length of mean w is " + str(len(np.mean(a, axis=0)))
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.mean(a, axis=0)
