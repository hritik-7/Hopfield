#import torch, torchvision
from PIL import Image
from os import listdir

from skimage.color import rgb2gray

from hopfield import ContinuousHopfield

import numpy as np
from PIL import Image
from os import listdir

from skimage.color import rgb2gray
from skimage.transform import resize

from matplotlib import pyplot as plt
from hopfield import ContinuousHopfield

import random

#Removes random chunk of data
def highBlocking(input, blockLevel):
    blocked = np.copy(input)

    for i in range(0, int(len(input)*blockLevel)):
        blocked[i] = 0
    return blocked

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

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

#Randomly inverts data
def randomFlipping(input, flipCount):
    flippy = np.copy(input)
    inv = np.random.binomial(n=1, p=flipCount, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            flippy[i] = random.uniform(0, 1)
    return flippy

#Removes random chunk of data
def highBlocking(input, blockLevel):
    blocked = np.copy(input)

    for i in range(0, int(len(input)*blockLevel)):
        blocked[i] = 0
    return blocked

#Invert full pattern
def fullFlipping(input):
    flippy = np.copy(input)
    return [-1 * flippy[i] for i in range(len(flippy))]

def preprocessing(img, dim=128):
    img = resize(img, (dim,dim), mode='reflect')
    #flatty = np.reshape(np.where(img>np.mean(img), 1, -1), (dim*dim))
    flatty = np.reshape(img, (dim*dim))
    return flatty

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
    plt.savefig("continuousResult.png")
    plt.show()





pics = []
for file in listdir("distinct"):
    foo = Image.open("distinct/"+file).convert("RGB")
    linear = preprocessing(rgb2gray(foo))
    pics.append(linear)


corrupted = [randomFlipping(d, 0.75) for d in pics]
#corrupted = [highBlocking(d, 0.4) for d in pics]

hoppy = ContinuousHopfield(pics)

predictions = []
for l in range(len(corrupted)):
    predictions.append(hoppy.predict(corrupted[l], 1))

    predictions[l] = [reshape(predictions[l][i]) for i in range(len(predictions[l]))]

pics = [reshape(pics[i]) for i in range(len(pics))]
resultsPlotter(pics, predictions)