#!/home/toddhu/anaconda3/bin/python
import numpy as np

'''                           Forward step of NN
                                                     By: Todd.Hu    2017.12.25
'''
set_of_input = 100

# Layer0: 4 nodes
num_of_layer0 = 4
z0 = np.random.rand(num_of_layer0) * 10
w0 = np.random.rand(num_of_layer0)


# Activation functions: reLU
def act_fun(arr):
    size_of_arr = arr.size
    print('size_of_arr =',size_of_arr)
    res = np.zeros(size_of_arr)
    print('initial res =',res)
    for ii in range(size_of_arr): 
        res[ii] = max(0,arr[ii])
        print('ii =',ii,'   res[ii]=',res)
    return(res)

# Layer1: 5 nodes
num_of_layer1 = 5
b1 = np.random.rand(num_of_layer1)
w1 = np.random.rand(num_of_layer1)
a1 = np.zeros(num_of_layer1)
z1 = np.zeros(num_of_layer1)

for i in range(num_of_layer1):
    a1[i] = np.dot(w0,z0) + b1[i]
    print('a1[i] = ',a1[i])
    z1[i] = act_fun(a1[i])
    print('z1[i] = ',z1[i])
    
#Layer2: 1 nodes
num_of_layer2 = 1
b2 = np.random.rand(num_of_layer2)
a2 = np.zeros(num_of_layer2)
z2 = np.zeros(num_of_layer2)

for i in range(num_of_layer2):
    a2[i] = np.dot(w1,z1) + b2[i]
    print('a2[i] = ',a2[i])
    z2[i] = act_fun(a2[i])
    print('z2[i] = ',z2[i])