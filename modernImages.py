#import torch, torchvision
import numpy as np
from PIL import Image
from os import listdir

from skimage.color import rgb2gray
from skimage.transform import resize

from matplotlib import pyplot as plt
from hopfield import ContinuousHopfield

import random

#Randomly inverts data
def randomFlipping(input, flipCount):
    flippy = np.copy(input)
    inv = np.random.binomial(n=1, p=flipCount, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            flippy[i] = random.uniform(0, 1)
    return flippy

#Invert full pattern
def fullFlipping(input):
    flippy = np.copy(input)
    return [-1 * flippy[i] for i in range(len(flippy))]

#Removes random chunk of data
def randomBlocking(input, blockLevel):
    blocked = np.copy(input)
    dim = int(np.sqrt(len(input)))

    xLoc = random.randint(0, int(dim - dim*blockLevel))
    yLoc = random.randint(0, int(dim - dim*blockLevel))

    for i in range(0, int(dim*blockLevel)):
        blocked[int((yLoc+i)*dim + xLoc): int(dim*blockLevel)] = 0
    return blocked

#Removes random chunk of data
def highBlocking(input, blockLevel):
    blocked = np.copy(input)

    for i in range(0, int(len(input)*blockLevel)):
        blocked[i] = 0
    return blocked

def preprocessing(img, dim=128):
    img = resize(img, (dim,dim), mode='reflect')
    #flatty = np.reshape(np.where(img>np.mean(img), 1, -1), (dim*dim))
    flatty = np.reshape(img, (dim*dim))
    print(flatty)
    return flatty

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def comparePatterns(pat1, pat2):
    #numpy.array_equal(a1, a2, equal_nan=False)
    valNormal = np.sum(pat1 == pat2)
    valFlipped = np.sum(pat1 == fullFlipping(pat2))
    if valNormal > valFlipped:
        print("Best normal")
        print("Amount same = ", valNormal/len(pat1))
    elif valNormal < valFlipped:
        print("Best flipped")
        print("Amount same = ", valFlipped/len(pat1))
    else:
        print("Amount same = ", valFlipped/len(pat1))

pics = []
for file in listdir("distinct"):
#for file in listdir("resizedFlowers"):
    #print(file)
    foo = Image.open("distinct/"+file).convert("RGB")
    #foo = Image.open("resizedFlowers/"+file).convert("RGB")

    #print(foo)
    linear = preprocessing(rgb2gray(foo))
    #print(linear)
    pics.append(linear)


corrupted = [randomFlipping(d, 0.95) for d in pics]
#corrupted = [highBlocking(d, 0.4) for d in pics]
#print(np.array(corrupted).shape)
#exit(1)
for l in range(len(corrupted)):
    print(np.min(corrupted[l]))
    print(np.max(corrupted[l]))
    print(np.min(pics[l]))
    print(np.max(pics[l]))
    #print(np.mean(pics[l]))
    comparePatterns(corrupted[l], pics[l])


hoppy = ContinuousHopfield(pics)

predictions = []
longest = 0
for l in range(len(corrupted)):
    predictions.append(hoppy.predict(corrupted[l], 2))

    #comparePatterns(predictions[l][0], pics[l])
    #comparePatterns(predictions[l][len(predictions[l])-1], pics[l])
    longest = max(longest, len(predictions[l]))

    predictions[l] = [reshape(predictions[l][i]) for i in range(len(predictions[l]))]


fig, axarr = plt.subplots(longest+1, len(predictions), figsize=(10, 10))
for l in range(len(predictions)):
    for i in range(len(predictions[l])+1):
        if i==0:
            axarr[i, l].imshow(reshape(pics[l]))
            axarr[i, l].axis('off')
        else:
            axarr[i, l].imshow(predictions[l][i-1])
            axarr[i, l].axis('off')

plt.tight_layout()
plt.savefig("result.png")
plt.show()