import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def entropy(S):
        S = np.array(S)
        epsilon = 1e-10  # Small constant to avoid division by zero
        S = S + epsilon
        log_probs = np.where(S > 0, np.log2(S), 0)
        ent = (-np.sum(S*log_probs))
        return ent

# Activations = [
#                 [0.0,0.0,0.0,1.0],
#                 [0.0,0.0,1.0,1.0],
#                 [0.0,1.0,1.0,1.0],
#                 [1.0,1.0,1.0,1.0],
#                 [0.0,0.8,0.9,1.0],
#                 [0.2,0.3,0.1,0.9],
#               ]


# Activations = [
#                 [0.0,0.0,0.0,1.0],
#                 [0.0,0.0,1.0,1.0],
#                 [0.0,1.0,1.0,1.0],
#                 [1.0,1.0,1.0,1.0],
#                 [0.0,0.8,0.9,1.0],
#                 [0.2,0.3,0.1,0.9],
#               ]

Activations = [ [0,1,0, 0],
                [0,0,0.25,1],
                [0,0.7,1,0],
                [0,0.3,1,0.3]
              ]
# Activations = np.random.rand(5,5)
# print(Activations)

linestyles = ['solid','dotted','dashed','dashdot','solid','dotted','dashed','dashdot']
colors = ['black','cyan','magenta','yellow','blue','red','orange','green']
count = 0
actv = []
sdev = []
for row in Activations:
# for rid in range(len(Activations)):
        # act = entropy(Activations[rid])
        act = entropy(row)
        # print(act)
        actv.append(act)
        sdev.append(np.exp(act))
print(Activations)
print(actv)
print(sdev)

plt.plot(sdev,label="Exp Entropy")
plt.plot(actv,label="Entropy")
plt.legend()
plt.xlabel("Standard Deviation")
plt.ylabel("Entropy")
plt.savefig("metric_test")
