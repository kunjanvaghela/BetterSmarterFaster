# Adjacency Matrix representation in Python
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from Code.Prey import Prey
from Code.Predator import Predator
from Code.Graph import Graph
from Code.customizedFunctions import *

import networkx as nx

size=50

# Variables
predator = Predator()
prey = Prey()
preyStateBelief, predatorStateBelief = {}, {}
pointSystem = {}
preyFoundCounter = 0
predatorFoundCounter = 0
states = []

# To initiate different state spaces for the graph
def initializeStates():
    # States represent unique state possible in the environment: (Agent Position, Prey, Predator)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                states.append((i,j,k))


if __name__=='__main__':
    g = Graph(size)
    start_time = datetime.datetime.now()

    initializeStates()
    print(states)
    print(len(states))


    # countTotal(res)


    end_time = datetime.datetime.now()
    print('Start time : '+str(start_time))
    print('End time : '+str(end_time))
    print('Total time : '+str(end_time-start_time))
