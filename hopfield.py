#Implementation of hopfield network
#General format idea adapted from pseudocode outline found here:
#https://www.geeksforgeeks.org/hopfield-neural-network/

import numpy as np

from torchvision import transforms
from scipy.special import softmax
from scipy.special import logsumexp

class Hopfield:
    #Initialisation function, gets weights for bipolar patterns
    def __init__(self, inputs):
        print("INITIALISING")

        self.n = len(inputs[0]) #no. of neurons
        self.itemsLen = len(inputs) # no. of patterns
        self.weights = np.empty((self.n,self.n)) #Connection matrix

        #w(ij) ​= ∑p​[s(i)​(p) * s(j)​(p)]
        for i in range(self.itemsLen):
            self.weights += np.outer(inputs[i], inputs[i])
        
        # notsure if this step is necessary
        self.weights = self.weights/self.itemsLen

    #Prediction function
    #yini ​​= xi​ + ∑j​[yj * ​wji​]
    #yi = yini{1 if > =0, else 0}
    #Iterates until hits iteration count or energy minimized
    def predict(self, input, iterations, theta = 0.0):
        print("Predictions")

        predicted = [np.copy(input)]

        s = self.energy(predicted[0])

        for i in range(0, iterations):
            #predicted.append(np.sign(self.weights @ predicted[i]))
            newVal = np.sign(predicted[i] + self.weights @ predicted[i])
            #predicted.append(np.sign(predicted[i] + self.weights @ predicted[i]))
            st = self.energy(newVal, theta = theta)
            if s == st:
                break
            s = st
            predicted.append(newVal)
        return predicted

    #E = 0.5 * ∑i​∑j[​wij​ * vi * ​vj] ​+ ∑i[​θi​ * vi]
    def energy(self, state, theta = 0.0):
        return -0.5 * state @ self.weights @ state + np.sum(state*theta)
    
import random

class DAMDiscreteHopfield:
    #based on 'Dense Associative Memory for Pattern Recognition' paper

    #Initialisation function
    def __init__(self, inputs):
        print("INITIALISING")

        self.n = len(inputs[0]) #no. of neurons
        self.N = len(inputs) # no. of patterns
        self.X = np.copy(inputs)
        
    
    #Update rule
    #Asynchronously flips all bits randomly
    #Keeps flipped bit if energy is lowered
    def predict(self, input, iterations = 5):
        print("Predictions")

        predicted = [np.copy(input)]
        
        for l in range(iterations):
            valList = np.arange(0, self.n)
            random.shuffle(valList)

            vals = predicted[l].copy()
            noFlip = True

            for i in valList:
                new_vals = vals.copy()
                new_vals[i] *= -1

                if (self.energy(new_vals) - self.energy(vals)) < 0:
                    vals[i] = new_vals[i]
                    noFlip = False

            predicted.append(vals)
            if noFlip:
                break
        return predicted
    
    #-∑F(state * X[i])
    def energy(self, state):
        x = np.array([(state*self.X[i]).sum() for i in range(len(self.X))])
        return -self.F(x, 2).sum()
    
    #F (x) = {if x > 0, x^n, else 0}
    def F(self, x, n):
        x[x < 0] = 0.
        return x**n



class ContinuousHopfield:
    #Initialisation function, gets weights for bipolar patterns
    def __init__(self, inputs):
        print("INITIALISING")

        self.n = len(inputs[0]) #no. of neurons
        self.itemsLen = len(inputs) # no. of patterns
        self.weights = np.empty((self.n,self.n)) #Connection matrix
        self.X = np.copy(inputs)
        #self.Xsum = []
        self.beta = 3
        self.logsumexp = logsumexp(self.X)
        print("logsumexp",self.logsumexp)
        for i in range(self.itemsLen):
            print("logsumexp",logsumexp(inputs[i]))

        #self.M = 

        #w(ij) ​= ∑p​[s(i)​(p) * s(j)​(p)] (where w(ij) ​= 0 for all i=j) 
        for i in range(self.itemsLen):
            self.weights += np.outer(inputs[i], inputs[i])
        
        # notsure if this step is necessary
        self.weights = self.weights/self.itemsLen

    # Defining the softmax function
    def softmax(self, values):
    
        print("values", values)
        # Computing element wise exponential value
        exp_values = np.exp(values)
        print("exp", exp_values)
    
        # Computing sum of these values
        exp_values_sum = np.sum(exp_values)

        print("exp_values_sum", exp_values_sum)
        print("exp_values/exp_values_sum", exp_values/exp_values_sum)

        # Returing the softmax output.
        return exp_values/exp_values_sum
    
    #Update rule
    def predict(self, input, iterations, theta = 0.0):
        print("Predictions")

        input = input/np.linalg.norm(input)

        predicted = [np.copy(input)]

        s = self.energy(predicted[0])

        normx = self.X/np.linalg.norm(self.X)
        normy = self.weights/np.linalg.norm(self.weights)
        
        #input = input/np.linalg.norm(input)

        for i in range(0, iterations):
            #print("NEW ROUND")
            #newVal = self.weights * softmax(self.beta * self.weights * predicted[i])
            #newVal = normx * softmax(self.beta * normx * predicted[i])
            #inner = self.beta * (self.X @ predicted[i])
            #print("ALL", normx * softmax(self.beta * normx * predicted[i]))
            #print("ALL", self.softmax(self.beta * (self.X @ predicted[i])))
            #print("ALL", self.softmax(self.beta * (self.X @ predicted[i])) @ self.X)
            #print(normy)


            #newVal = self.softmax(self.beta * (self.X @ predicted[i])) @ self.X
            #newVal = softmax(self.beta * (self.X @ predicted[i])) @ self.X

            #newVal = softmax(self.beta * (predicted[i] @ np.transpose(normy))) @ normy

            newVal = softmax(self.beta * (np.transpose(normy) @ predicted[i])) @ normy


            #print(self.call(input) )
            #print("1",softmax(self.beta * (predicted[i] @ np.transpose(normy))))
            
            #print("1",softmax(self.beta * (self.X @ predicted[i])))
            #print("2",softmax((predicted[i] @ np.transpose(self.X))))

            #for l in range(0, self.itemsLen):
                #print(np.exp(self.X[l] * self.beta))
            #    print("COMP",self.X[l] * predicted[i], np.sum(self.X[l] * predicted[i]))

            #print("LSE", self.LSE(newVal @ np.transpose(self.X)))
            #print("LSE", self.LSE(self.X @ newVal))
            #print("ENERGY", self.energy(newVal))

            #newVal = self.X[0] * self.softmax(self.beta * self.X[0] * predicted[i])
            #print("MINI", self.softmax(self.beta * self.X[0] * predicted[i]))
            #for p in range(1, self.itemsLen):
            #    newVal += self.X[p] * self.softmax(self.beta * self.X[p] * predicted[i])
                #print("MINI", self.softmax(self.beta * self.X[p] * predicted[i]))
            #st = self.energy(newVal, theta = theta)
            #if s == st:
            #    break
            #s = st
            predicted.append(newVal)
        return predicted
    
    def LSE(self, x):
        inner = np.exp(x[0] * self.beta)
        for l in range(1, self.itemsLen):
            inner += np.exp(x[l] * self.beta)
        return pow(self.beta, -1) * np.log(inner)
        #return pow(self.beta, -1) * np.log(np.sum(np.exp(self.beta * x)))
    
    #E=−∑[exp(x^T_i * ξ)]
    #F is function F(z) = z^a
    #N stored patterns as {xi}Ni=
    #state pattern ξ
    def energy(self, state):
        return -self.LSE(self.X @ state) + 0.5 * state*state + pow(self.beta, -1) * np.log(self.itemsLen) + 0.5*self.n*self.n