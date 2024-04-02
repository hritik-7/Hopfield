#Implementation of hopfield network
#General format idea adapted from pseudocode outline found here:
#https://www.geeksforgeeks.org/hopfield-neural-network/

import numpy as np

from torchvision import transforms

class Hopfield:
    #Initialisation function, gets weights for bipolar patterns
    def __init__(self, inputs):
        print("INITIALISING")

        self.n = len(inputs[0]) #no. of neurons
        self.itemsLen = len(inputs) # no. of patterns
        self.weights = np.empty((self.n,self.n)) #Connection matrix

        #w(ij) ​= ∑p​[s(i)​(p) * s(j)​(p)] (where w(ij) ​= 0 for all i=j) 
        for i in range(self.itemsLen):
            self.weights += np.outer(inputs[i], inputs[i])
        
        # notsure if this step is necessary
        self.weights = self.weights/self.itemsLen

    #Prediction function
    #yini ​​= xi​ + ∑j​[yj * ​wji​]
    #yi = yini{1 if > =0, else 0}
    #Iterates until hits iteration count or energy minimized
    def predict(self, input, iterations):
        print("Predictions")

        predicted = [np.copy(input)]

        s = self.energy(predicted[0])

        for i in range(0, iterations):
            #predicted.append(np.sign(self.weights @ predicted[i]))
            newVal = np.sign(predicted[i] + self.weights @ predicted[i])
            #predicted.append(np.sign(predicted[i] + self.weights @ predicted[i]))
            st = self.energy(newVal)
            if s == st:
                break
            s = st
            predicted.append(newVal)
        return predicted

    #E = 0.5 * ∑i​∑j[​wij​ * vi * ​vj] ​+ ∑i[​θi​ * vi]
    #Haven't added thresholding
    def energy(self, state):
        return -0.5 * state @ self.weights @ state