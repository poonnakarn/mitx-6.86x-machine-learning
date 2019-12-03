import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
for i in range(1,5):
    costs = []
    for j in range(5):
        mixture, post = common.init(X, i, j)
        _, _, cost = kmeans.run(X, mixture, 0)
        costs.append(cost)
        common.plot(X, mixture,post, 'test')
    print(min(costs))
        
