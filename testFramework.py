from hopfield import Hopfield, ContinuousHopfield, DAMDiscreteHopfield
import numpy as np
import random

#Randomly inverts data
def randomFlipping(input, flipCount):
    flippy = np.copy(input)
    inv = np.random.binomial(n=1, p=flipCount, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            flippy[i] = -1 * v
    return flippy

#Removes random chunk of data
def highBlocking(input, blockLevel):
    blocked = np.copy(input)

    for i in range(0, int(len(input)*blockLevel)):
        blocked[i] = -1
    return blocked



print("============================================")
print("Hopfield")
print("============================================")

for i in range(40, 800, 40):
    patterns = np.array([random.choices([-1,1], k=1000) for l in range(i)])
    hoppy = Hopfield(patterns)

    corrupted = [highBlocking(d, 0.4) for d in patterns]

    predictions = []
    for l in range(len(corrupted)):
        predictions.append(hoppy.predict(corrupted[l], 3)[-1])
    
    print(i, ":", (patterns==predictions).sum()/(1000*i))


print("============================================")
print("Dense Associative Memory")
print("============================================")

#for i in range(40, 800, 40):
for i in range(2, 40, 4):
    patterns = np.array([random.choices([-1,1], k=128) for l in range(i)])
    hoppy = DAMDiscreteHopfield(patterns)

    corrupted = [highBlocking(d, 0.4) for d in patterns]

    predictions = []
    for l in range(len(corrupted)):
        predictions.append(hoppy.predict(corrupted[l], 3)[-1])
    
    print(i, ":", (patterns==predictions).sum()/(1000*i))