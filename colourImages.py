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
            #flippy[i] = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
            #flippy[i][0] = random.uniform(0, 1)
            #flippy[i][1] = random.uniform(0, 1)
            #flippy[i][2] = random.uniform(0, 1)
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
    print(img.shape)
    flatty = np.reshape(img, (dim*dim, 3))
    print(flatty)
    return flatty

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim, 3))
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

    print(type(foo))
    print(type(rgb2gray(foo)))

    linear = preprocessing(np.array(foo))
    #linear = preprocessing(rgb2gray(foo))
    pics.append(linear)


band0 = []
band1 = []
band2 = []
for i in range(len(pics)):
    band0.append([pics[i][l][0] for l in range(len(pics[i]))])
    band1.append([pics[i][l][1] for l in range(len(pics[i]))])
    band2.append([pics[i][l][2] for l in range(len(pics[i]))])
band0 = np.array(band0)
band1 = np.array(band1)
band2 = np.array(band2)

print(band0.shape)
corrupted0 = [randomFlipping(d, 0.95) for d in band0]
corrupted1 = [randomFlipping(d, 0.95) for d in band1]
corrupted2 = [randomFlipping(d, 0.95) for d in band2]

#for l in range(len(corrupted0)):
#    comparePatterns(corrupted0[l], pics[l])


hoppy0 = ContinuousHopfield(band0)
hoppy1 = ContinuousHopfield(band1)
hoppy2 = ContinuousHopfield(band2)

predictions = []
longest = 1

for l in range(len(corrupted0)):
    predictionsa = hoppy0.predict(corrupted0[l], 2)
    predictionsb = hoppy1.predict(corrupted1[l], 2)
    predictionsc = hoppy2.predict(corrupted2[l], 2)

    #longest = max(longest, len(predictions[l]))

    newPred = []
    for i in range(len(predictionsa)):
        newPic = []
        for l in range(len(predictionsa[i])):
            newPic.append([predictionsa[i][l], predictionsb[i][l], predictionsc[i][l]])
        newPred.append(np.copy(np.array(newPic)))

    predictions.append(newPred)
    #predictions.append([[predictionsa[i], predictionsb[i], predictionsc[i]] for i in range(len(predictionsa))])
    #predictions[l] = [reshape(predictions[l][i]) for i in range(len(predictions[l]))]

predictions = np.array(predictions)
print("predictions.shape",predictions.shape)
print("pics.shape",np.array(pics).shape)
#print("predictions",predictions)
#print("pics",pics)

#predictions = reshape(predictions)

fig, axarr = plt.subplots(3+1, len(pics), figsize=(10, 10))
for l in range(len(pics)):
    axarr[0, l].imshow(reshape(pics[l]))
    axarr[0, l].axis('off')


for l in range(0, len(predictions)):
    for i in range(1, len(predictions[l])+1):
        #axarr[i, l].imshow(predictions[l][i-1])
        axarr[i, l].imshow(reshape(predictions[l][i-1]))
        axarr[i, l].axis('off')


plt.tight_layout()
plt.savefig("result.png")
plt.show()