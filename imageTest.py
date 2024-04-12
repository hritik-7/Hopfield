#Hopfield and DAM class

import numpy as np
from PIL import Image
from os import listdir

from skimage.color import rgb2gray
from skimage.transform import resize

from matplotlib import pyplot as plt
from hopfield import *

import random
#from helper import *

import numpy as np
from PIL import Image
from os import listdir

from skimage.color import rgb2gray
from skimage.transform import resize

from matplotlib import pyplot as plt
from hopfield import Hopfield
from hopfield import DAMDiscreteHopfield

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
def randomBlocking(input, blockLevel):
    blocked = np.copy(input)
    dim = int(np.sqrt(len(input)))

    xLoc = random.randint(0, int(dim - dim*blockLevel))
    yLoc = random.randint(0, int(dim - dim*blockLevel))

    for i in range(0, int(dim*blockLevel)):
        blocked[int((yLoc+i)*dim + xLoc): int(dim*blockLevel)] = -1
    return blocked

#Removes random chunk of data
def highBlocking(input, blockLevel):
    blocked = np.copy(input)

    for i in range(0, int(len(input)*blockLevel)):
        blocked[i] = -1
    return blocked

def preprocessing(img, dim=128):
    img = resize(img, (dim,dim), mode='reflect')
    flatty = np.reshape(np.where(img>np.mean(img), 1, -1), (dim*dim))
    return flatty

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def comparePatterns(pat1, pat2):
    valNormal = np.sum(pat1 == pat2)
    valFlipped = np.sum(pat1 == pat2 * -1)
    if valNormal > valFlipped:
        print("Amount same = ", valNormal/len(pat1))
        return valNormal/len(pat1)
    elif valNormal < valFlipped:
        print("Amount same = ", valFlipped/len(pat1))
        return valFlipped/len(pat1)
    else:
        print("Amount same = ", valFlipped/len(pat1))
        return valFlipped/len(pat1)

def getAccuracy(originals, finalised):
    correct = 0
    for i in range(len(originals)):
        valNormal = np.sum(finalised[i][-1] == originals[i])
        valFlipped = np.sum(finalised[i][-1] == originals[i] * -1)
        if valNormal == 1 or valFlipped == 0:
            correct += 1
    return correct/len(originals)

def resultsPlotter(original, iterations):
    longest  = 1
    for i in range(len(iterations)):
        longest = max(longest, len(iterations[i])+1)
    fig, axarr = plt.subplots(len(original), longest, figsize=(10, 10))
    axarr[0, 0].set_title('Originals')
    axarr[0, 1].set_title('Corrupted')

    for l in range(0, len(original)):
        for i in range(0, longest):
            axarr[l, i].axis('off')


    for l in range(0, len(original)):
        axarr[l, 0].imshow(original[l])
        axarr[l, 0].axis('off')


    for l in range(0, len(iterations)):
        for i in range(0, len(iterations[l])):
            axarr[0, i+1].set_title('Iteration ' +str(i))
            axarr[l, i+1].imshow(iterations[l][i])
            axarr[l, i+1].axis('off')


    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()






print("Loading images")
picNumber = 0
pics = []
for file in listdir("distinct"):
    foo = Image.open("distinct/"+file).convert("RGB")

    linear = preprocessing(rgb2gray(foo))
    pics.append(linear)
    picNumber+=1
    if picNumber > 5:
        break

print("Corrupting Images")

#corrupted = [randomFlipping(d, 0.4) for d in pics]
corrupted = [highBlocking(d, 0.4) for d in pics]
           
hoppy = Hopfield(pics)
#hoppy = DAMDiscreteHopfield(pics)


predictions = []
longest = 0

print("Running hopfield")
for l in range(len(corrupted)):
    predictions.append(hoppy.predict(corrupted[l], 3))

    comparePatterns(predictions[l][-1], pics[l])
    longest = max(longest, len(predictions[l]))

    predictions[l] = [reshape(predictions[l][i]) for i in range(len(predictions[l]))]

#print(getAccuracy(pics, predictions))
pics = [reshape(pics[i]) for i in range(len(pics))]
resultsPlotter(pics, predictions)