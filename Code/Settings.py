from Code.Prey import Prey
from Code.Predator import Predator
from Code.Graph import Graph
from Code.customizedFunctions import *

# Variables
size = 50
predator = Predator()
prey = Prey()
agentPos = -1
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