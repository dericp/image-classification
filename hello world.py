import numpy as np
def list2array(list):
    input_2darray = np.zeros([5000,401])
    for i in range(len(list)):
        token = list[i].split(' ')
        for j in range(len(token)):
            input_2darray[i,j] = float(token[j])
    return input_2darray
