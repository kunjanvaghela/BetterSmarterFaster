import numpy as np
from activations import *



class NeuralNetwork:
    def __init__(self,layerDimensions):
        self.params = self.initializeParameters(layerDimensions)

    def training(self,X, Y, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
        np.random.seed(2)
        rewards = []
        for i in range(0, num_iterations):
            AL, stashes = linearModelForward(X, self.params)
            reward = computedCost(AL, Y)
            grads = linearModelBackward(AL, Y, stashes)
            self.params = updateParameters(self.params, grads, learning_rate)
            if print_cost and i % 50 == 0:
                print ("Reward after iteration %i: %f" %(i, reward))
                rewards.append(reward)
        plt.plot(np.squeeze(rewards))
        plt.ylabel('Reward')
        plt.xlabel('Iterations (per 100s)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    def initializeParameters(self,layerDimensions):
        np.random.seed(1)
        parameters = {}
        L = len(layerDimensions)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layerDimensions[l], layerDimensions[l-1]) / np.sqrt(layerDimensions[l-1])*0.1
            parameters['b' + str(l)] = np.zeros((layerDimensions[l], 1))
        return parameters

    def predict(self,X, y):
        m = X.shape[1]
        n = len(self.params) // 2
        p = np.zeros((1,m))
        probas,_ = linearModelForward(X, self.params)
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
        print("Accuracy: "  + str(np.sum((p == y)/m)))
        return p
