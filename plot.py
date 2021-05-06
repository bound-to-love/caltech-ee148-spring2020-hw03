import matplotlib.pyplot as plt 
import numpy as np

'''
Full train set: train: 99%; test: 99%
50% train set: train: 98%; test: 98%
25% train set: train: 98%; test: 98%
12.5% train set: train: 97%; test: 97%
6.25% train set: train: 95%; test: 94%
'''

set_size = np.log([1.,.5,.25,.125,.0625])
train_acc = np.log([.99,.98,.98,.97,.95])
test_acc = np.log([.99,.98,.98,.97,.94])

ptrain,=plt.plot(set_size, train_acc, c='g')
ptest,=plt.plot(set_size, test_acc, c='r')
plt.xlabel("log fraction train set size")
plt.ylabel("log acurracy")
plt.legend([ptrain,ptest],['train','test'])
plt.show()
