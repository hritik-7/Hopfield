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

            if noFlip:
                break
            predicted.append(vals)
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
    #based on 'Dense Associative Memory for Pattern Recognition' paper

    #Initialisation function
    def __init__(self, inputs):
        print("INITIALISING")

        self.n = len(inputs[0]) #no. of neurons
        #self.M = len(inputs[0]) #Max Length (Probably)
        self.M = np.linalg.norm(inputs[0])
        self.N = len(inputs) # no. of patterns
        self.X = np.copy(inputs)
        
    
    #Update rule
    #Asynchronously flips all bits randomly
    #Keeps flipped bit if energy is lowered
    def predict(self, input, iterations = 5, beta = 8):
        print("Predictions")

        #input = input/np.linalg.norm(input)
        predicted = [np.copy(input)]
        energy = self.energy(input, beta)

        #normx = self.X/np.linalg.norm(self.X)
        
        for l in range(iterations):
            print("SOFTY",softmax(np.array([np.sum(predicted[l]*self.X[i]) for i in range(len(self.X))])))
            vals = softmax(np.array([np.sum(predicted[l]*self.X[i]) for i in range(len(self.X))])) @ self.X

            #vals = softmax(np.array([np.sum((predicted[l]-self.X[i])**2) for i in range(len(self.X))])) @ self.X
            
            #vals = softmax(np.array([np.sum(predicted[l]*normx[i]) for i in range(len(self.X))])) @ self.X
            #vals = softmax(beta * (np.transpose(self.X) @ input)) @ self.X

            #valList = np.arange(0, self.n)
            #random.shuffle(valList)




            #vals = predicted[l].copy()
            #noFlip = True

            #for i in valList:
            #    new_vals = vals.copy()
            #    new_vals[i] *= -1

            #    if (self.energy(new_vals) - self.energy(vals)) < 0:
            #        vals[i] = new_vals[i]
            #        noFlip = False

            #if noFlip:
            #    break
            new_energy = self.energy(vals, beta)
            #if not new_energy < energy:
            #    break
            print("ENERGY", new_energy, energy, new_energy< energy, 2 * self.M**2, self.energy(vals, beta) < 2 * self.M**2)
            predicted.append(vals)
        return predicted
    
    def LSE(self, beta, X):
        return np.log(np.sum([np.exp(beta*X[i]) for i in range(len(X))])) / beta
    
    #-∑F(state * X[i])
    def energy(self, state, beta):
        print("0.5*np.transpose(state)*state", 0.5*np.transpose(state)@state)
        print("np.log(self.N)/beta", np.log(self.N)/beta)
        print("0.5 * self.M**2", 0.5 * self.M**2)
        #lse = -self.LSE(beta, np.array([state*self.X[i] for i in range(len(self.X))]))
        
        #print("np.transpose(state)@state",np.transpose(self.X[0])@state)
        #print("state * state",self.X[0] * state)
        #print("np.exp(beta * self.X[i] * state)", np.exp(beta * self.X[0] * state))
        
        #lse = -np.log(np.sum([np.exp(beta * self.X[i] * state) for i in range(len(self.X))])) / beta
        lse = -np.log(np.sum([np.exp(beta * self.X[i] * state) for i in range(len(self.X))])) / beta
        print("lse",lse)
        #x = -self.LSE(beta, np.transpose(self.X) ) + 0.5*np.transpose(state)@state + np.log(self.N)/beta + 0.5 * self.M**2
        x = lse + 0.5*np.transpose(state)@state + np.log(self.N)/beta + 0.5 * self.M**2
        return x