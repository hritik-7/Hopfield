#Implementation of hopfield network
#General format idea adapted from pseudocode outline found here:
#https://www.geeksforgeeks.org/hopfield-neural-network/

import numpy as np

from scipy.special import softmax
#from scipy.special import logsumexp
import random

class Hopfield:
    #Initialisation function, gets weights for bipolar patterns
    def __init__(self, inputs):

        self.n = len(inputs[0]) #no. of neurons
        self.itemsLen = len(inputs) # no. of patterns
        self.weights = np.empty((self.n,self.n)) #Connection matrix
        self.X = np.array(inputs)

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
        #print("Predictions")

        predicted = [np.copy(input)]

        s = self.energy(predicted[0])

        for i in range(0, iterations):
            newVal = np.sign(self.weights @ predicted[i])

            st = self.energy(newVal, theta = theta)
            if s == st:
                break
            s = st
            predicted.append(newVal)
        return predicted

    #E = 0.5 * ∑i​∑j[​wij​ * vi * ​vj] ​+ ∑i[​θi​ * vi]
    def energy(self, state, theta = 0.0):
        return -0.5 * state @ self.weights @ state + np.sum(state*theta)
    
class DAMDiscreteHopfield:
    #based on 'Dense Associative Memory for Pattern Recognition' paper

    #Initialisation function
    def __init__(self, inputs):
        self.n = len(inputs[0]) #no. of neurons
        self.N = len(inputs) # no. of patterns
        self.X = np.copy(inputs)
        
    
    #Update rule
    #Asynchronously flips all bits randomly
    #Keeps flipped bit if energy is lowered
    def predict(self, input, iterations = 5):

        predicted = [np.copy(input)]
        
        for l in range(iterations):
            valList = np.arange(0, self.n)
            random.shuffle(valList)


            vals = predicted[l].copy()
            noFlip = True

            prev = self.energy(vals)

            for i in valList:
                new_vals = vals.copy()
                new_vals[i] *= -1

                current = self.energy(new_vals)

                if (current - prev) < 0:
                    prev = self.energy(new_vals)
                    vals[i] = new_vals[i]
                    noFlip = False
            if noFlip:
                break
            predicted.append(vals)
        return predicted
    
    #-∑F(state * X[i])
    def energy(self, state):
        x = self.X@state
        return -self.F(x, 2).sum()
    
    #F (x) = {if x > 0, x^n, else 0}
    def F(self, x, n):
        x[x < 0] = 0.
        return x**n

#Continuous Hopfield
#Based on:
#Hopfield Networks is All You Need
class ContinuousHopfield:
    #based on 'Dense Associative Memory for Pattern Recognition' paper

    #Initialisation function
    def __init__(self, inputs):
        self.n = len(inputs[0]) #no. of neurons
        self.M = np.linalg.norm(inputs[0])
        self.N = len(inputs) # no. of patterns
        self.X = np.copy(inputs)

        newX = np.copy(self.X)
        self.newX = np.array([newX[i]/np.mean(newX[i]) for i in range(len(newX))])
        
    
    #Update rule
    #X softmax(beta X^T ξ)
    def predict(self, input, iterations = 1, beta = 8):
        predicted = [np.copy(input)]
        #energy = self.energy(input, beta)

        for i in range(iterations):
            vals = softmax(beta * predicted[i] @ np.transpose(self.newX) ) @ self.X 
            #vals = softmax(beta * input @ np.transpose(self.X) ) @ self.X 
        
            #new_energy = self.energy(vals, beta)
            #if not new_energy < energy:
            #    break
            #print("ENERGY", new_energy, energy, new_energy< energy, 2 * self.M**2, self.energy(vals, beta) < 2 * self.M**2)
            
            #if vals == predicted[i]:
            #    break
            predicted.append(vals)
        return predicted
    
    # log(∑i[exp(βxi)])/β
    def LSE(self, beta, X):
        return np.log(np.sum([np.exp(beta*X[i]) for i in range(len(X))])) / beta
    
    #Energy rule
    # E = − lse(β, X^T ξ) + 0.5 * ξ^T ξ + log(N)/β + 0.5 * M^2   
    def energy(self, state, beta):
        lse = -np.log(np.sum([np.exp(beta * self.X[i] * state) for i in range(len(self.X))])) / beta
        x = lse + 0.5*np.transpose(state)@state + np.log(self.N)/beta + 0.5 * self.M**2
        return x