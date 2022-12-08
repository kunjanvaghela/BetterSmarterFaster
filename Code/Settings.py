from Code.Prey import Prey
from Code.Predator import Predator
from Code.Graph import Graph
from Code.customizedFunctions import *

# Variables
size = 50
predator = Predator()
prey = Prey()
agentPos = -1
agentIterations = 10000
g = Graph(size)
preyStateBelief, predatorStateBelief = {}, {}
pointSystem = {}
preyFoundCounter = 0
predatorFoundCounter = 0
states = []
utilityOfStates = {}
utilityAtTimeT = {}
utilityAtTimeTMinus1 = {}
probabilityStateTransition = {}
utilityOfNextAction = {}

# Rewards
# rewardPrey = float(1)
# rewardPredator = - float(1000000)
# rewardNormal = - float(1)

rewardPrey = float(0)
rewardPredator = - float('inf')
rewardNormal = - float(1)