from Code.Prey import Prey
from Code.Predator import Predator
from Code.Graph import Graph
from Code.customizedFunctions import *

# Variables
size = 50
predator = Predator()
prey = Prey()
agentPos = -1
agentIterations = 1000
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
dataForUPartialAgent = []
newEnv = 1
vModel = 1   #Or reading U* from existing file.
# fileNameToReadUtility = 'UtilitiesFinal.csv'
# fileNameToOutput = 'Agent_UStar_Result.csv'
# fileNameToReadUtility = 'VModelOutput2_NoIndex.csv'
# fileNameToOutput = 'Agent_VModel_Result.csv'
fileNameToReadUtility = 'UtilitiesFinal.csv'
fileNameToOutput = 'Agent_UPartial_Result.csv'

# Rewards
# rewardPrey = float(1)
# rewardPredator = - float(1000000)
# rewardNormal = - float(1)

rewardPrey = float(0)
rewardPredator = - float('inf')
rewardNormal = - float(1)