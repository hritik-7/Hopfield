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
def highBlocking(input, blockLevel):
    blocked = np.copy(input)
    for i in range(0, int(len(input)*blockLevel)):
        blocked[i] = 0
    return blocked

def preprocessing(img, dim=128):
    img = resize(img, (dim,dim), mode='reflect')
    flatty = np.reshape(img, (dim*dim, 3))
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


def resultsPlotter(original, iterations):
    fig, axarr = plt.subplots(len(original), len(iterations[0])+1, figsize=(10, 10))
    axarr[0, 0].set_title('Originals')
    axarr[0, 1].set_title('Corrupted')


    for l in range(0, len(original)):
        axarr[l, 0].imshow(reshape(original[l]))
        axarr[l, 0].axis('off')


    for l in range(0, len(iterations)):
        for i in range(0, len(iterations[l])):
            axarr[0, i+1].set_title('Iteration ' +str(i))
            #axarr[i, l].imshow(predictions[l][i-1])
            axarr[l, i+1].imshow(reshape(iterations[l][i]))
            axarr[l, i+1].axis('off')


    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

print("Loading Images")
pics = []
for file in listdir("distinct"):
    foo = Image.open("distinct/"+file).convert("RGB")

    linear = preprocessing(np.array(foo))
    pics.append(linear)


print("Corrupting Images")
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

corrupted0 = [randomFlipping(d, 0.85) for d in band0]
corrupted1 = [randomFlipping(d, 0.85) for d in band1]
corrupted2 = [randomFlipping(d, 0.85) for d in band2]



hoppy0 = ContinuousHopfield(band0)
hoppy1 = ContinuousHopfield(band1)
hoppy2 = ContinuousHopfield(band2)

predictions = []
longest = 1

print("Running hopfield")
for l in range(len(corrupted0)):
    predictionsa = hoppy0.predict(corrupted0[l], 1)
    predictionsb = hoppy1.predict(corrupted1[l], 1)
    predictionsc = hoppy2.predict(corrupted2[l], 1)


    newPred = []
    for i in range(len(predictionsa)):
        newPic = []
        for l in range(len(predictionsa[i])):
            newPic.append([predictionsa[i][l], predictionsb[i][l], predictionsc[i][l]])
        newPred.append(np.copy(np.array(newPic)))

    predictions.append(newPred)

predictions = np.array(predictions)
resultsPlotter(pics, predictions)