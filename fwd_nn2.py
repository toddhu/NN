
# coding: utf-8

# In[48]:

#!/home/toddhu/anaconda3/bin/python
import numpy as np

'''                           Forward step of NN
                                                     By: Todd.Hu    2017.12.25
'''
num_of_set = 10

# Layer0: 4 nodes
num_of_layer0 = 4
z0 = np.random.rand(num_of_set,num_of_layer0) * 10
w0 = np.random.rand(num_of_layer0)

print('***** number of set =',num_of_set,'*****\ninitial X0 =\n',z0,'\ninitial w0 =',w0)

# Activation functions: reLU
def act_fun(arr):
    size_of_arr = arr.size
    print('size_of_arr =',size_of_arr)
    res = np.zeros(size_of_arr)
    print('initial res =',res)
    for i in range(size_of_arr): 
        res[i] = max(0,arr[i])
        print('i =',i,'   res[i]=',res)
    return(res)

# Layer1: 5 nodes
num_of_layer1 = 4
b1 = np.random.rand(num_of_layer1)
w1 = np.random.rand(num_of_layer1)
a1 = np.zeros(num_of_layer1)
z1 = np.zeros((num_of_set,num_of_layer1))

#Layer2: 3 nodes
num_of_layer2 = 3
b2 = np.random.rand(num_of_layer2)
a2 = np.zeros(num_of_layer2)
z2 = np.zeros((num_of_set,num_of_layer2))

print('initial b1 =',b1,'\ninitial w1 =',w1,'\ninitial a1 =',a1,'\ninitial z1 =',z1)
print('initial b2 =',b2,'\ninitial a2 =\n',a2,'\ninitial z2 =\n',z2)
print("----------------------------------\nHere We Start . Let's go!\n=======================================>>")
for seris in range(num_of_set):
    print('\nnumber of set =',seris,'***********************************************************\n')
    for i in range(num_of_layer1):
        a1[i] = np.dot(w0,z0[seris]) + b1[i]
        print('i =',i,'   a1[i] = ',a1[i])
    
    z1[seris]= act_fun(a1)
    print('\nin seris =',seris,'=============================================================>\n z1 = \n',z1)
 
    for i in range(num_of_layer2):
        a2[i] = np.dot(w1,z1[seris]) + b2[i]
        print('i =',i,'   a2[i] = ',a2[i])
    z2[seris] = act_fun(a2)
    print('\nin seris =',seris,'=============================================================>\n z2 =\n ',z2)
print('                                   ### [The forward step has been Done]###')


# In[ ]:



