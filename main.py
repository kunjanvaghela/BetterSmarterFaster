# Adjacency Matrix representation in Python
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
# from Code.Prey import Prey
# from Code.Predator import Predator
# from Code.Graph import Graph
from Code.customizedFunctions import *
import Code.Settings as gVar



# To initiate different state spaces for the graph
def initializeStates():
    # States represent unique state possible in the environment: (Agent, Prey, Predator) positions
    for i in range(gVar.size):
        for j in range(gVar.size):
            for k in range(gVar.size):
                gVar.states.append((i,j,k))

# To initialize reward of being in the current state
def initializeRewardOfStateS():
    # Initializes reward of being in a particular state
    for s in gVar.states:
        gVar.utilityOfStates[s] = {}
        reward = float(-1)
        if s[0] == s[1]:
            reward = float(10000)           # Reward of being with the prey
        if s[0] == s[2]:
            reward = - float(10000)      # Reward of being with the predator
        gVar.utilityOfStates[s][0] = reward


def getNextStates(stateS):
    nextStates = set()      # To store next states reachable from stateS in the var nextStates
    # To get next agent, prey and predator positions from the current agent, prey and predator positions
    nextAgentPos = gVar.g.getNextNodes(stateS[0])
    nextAgentPos.append(stateS[0])
    nextPreyPos = gVar.g.getNextNodes(stateS[1])
    nextPreyPos.append(stateS[1])
    nextPredatorPos = gVar.g.getNextNodes(stateS[2])
    nextPredatorPos.append(stateS[2])
    for i in nextAgentPos:
        for j in nextPreyPos:
            for k in nextPredatorPos:
                nextStates.add((i,j,k))
    return nextStates


# To set utility of stateS at timeT
def probabilityOfNextState(stateS, timeT):
    # Get probability of all the next cells of the predator
    nextPredatorPos = gVar.g.getNextNodes(stateS[2])
    probPredator = {i:float(0) for i in nextPredatorPos}
    bfsResult = gVar.g.breadthFirstSearch(stateS[2], stateS[0])[0]  # Gets the path from Predator to Agent positions
    if len(bfsResult) == 1:
        # If Predator and Agent Positions are the same, then bfs path will be 1 only. Adding current node in nextPredatorPos list.
        probPredator[stateS[2]] = float(0)
        nextPredatorPos.append(stateS[2])
    for nextPred in nextPredatorPos:
        if len(bfsResult) > 1:
            # Adding 0.6 probability of predator going to the next node lying in BFS path to Agent
            if nextPred == bfsResult[1]:
                probPredator[nextPred] += 0.6
        elif len(bfsResult) == 1:
            # Since Predator will be in same cell only in this condition, adding 0.6 probability to the current cell of the predator.
            if nextPred == bfsResult[0]:
                probPredator[nextPred] += 0.6
        # Divinding 0.4 probability into all other nodes that predator can go to
        probPredator[nextPred] += ((1 - 0.6) / len(nextPredatorPos))
    
    # Calculate probability of Prey going to all the next cells from the current Prey Position
    nextPreyPos = gVar.g.getNextNodes(stateS[1])
    nextPreyPos.append(stateS[1])
    probPrey = {i:float(0) for i in nextPreyPos}
    for nextPrey in nextPreyPos:
        probPrey[nextPrey] += (1 / len(nextPreyPos))

    # Calculate timestamp t's Utility
    nextStates = getNextStates(stateS)
    for nextState in nextStates:
        if nextState[1] in probPrey and nextState[2] in probPredator:
            if gVar.utilityOfStates[stateS][timeT - 1] == 10000 or gVar.utilityOfStates[stateS][timeT - 1] == -float(10000):
                gVar.utilityOfStates[stateS][timeT] = -float('inf')
                continue
            mid = gVar.utilityOfStates[stateS][0] + (gVar.utilityOfStates[nextState][timeT-1] * (probPrey[nextState[1]] * probPredator[nextState[2]]))
            if gVar.utilityOfStates[stateS][timeT] < mid:
                gVar.utilityOfStates[stateS][timeT] = mid


def calculateUtility():
    initializeRewardOfStateS()
    # print('rewardOfStateS : ',gVar.utilityOfStates)
    # for i in range(1,gVar.utilityIterationCount + 1):
    #     for stateS in gVar.states:
    #         gVar.utilityOfStates[stateS][i] = - float('inf')
    statesToCheck = gVar.states
    i = 1
    while True:
        print('Calculation for i = ',i)
        for stateS in statesToCheck:
            gVar.utilityOfStates[stateS][i] = - float('inf')
            # if gVar.utilityOfStates[stateS][i - 1] != -float('inf'):     # This is to implement that no need to calculate if prey or predator found in earlier timestamp.
            probabilityOfNextState(stateS, i)
        toRemove = []
        if i < 4:
            for stateS in statesToCheck:
                if gVar.utilityOfStates[stateS][i] == -float('inf'):
                    # Remove from statesToCheck
                    toRemove.append(stateS)
                    # statesToCheck.remove(stateS)
        elif i > 5:
            for stateS in statesToCheck:
                first = gVar.utilityOfStates[stateS][i]
                second = gVar.utilityOfStates[stateS][i-1]
                third = gVar.utilityOfStates[stateS][i-2]
                first = abs(first * 1000000)
                second = abs(second * 1000000)
                third = abs(third * 1000000)
                if first == second and first == third:
                    # Remove from statesToCheck
                    toRemove.append(stateS)
                    # statesToCheck.remove(stateS)
        for r in toRemove:
            statesToCheck.remove(r)
        if len(statesToCheck) == 0:
            break
        i += 1
    # for i in range(1,gVar.utilityIterationCount + 1):
    #     print('Calculation for i = ',i)
    #     for stateS in gVar.states:
    #         gVar.utilityOfStates[stateS][i] = - float('inf')
    #     for stateS in gVar.states:
    #         if gVar.utilityOfStates[stateS][i - 1] != -float('inf'):     # This is to implement that no need to calculate if prey or predator found in earlier timestamp.
    #             probabilityOfNextState(stateS, i)

# To get max value from the utility function 
def getMaxValueFromState(stateS):
    values = gVar.utilityOfStates[stateS]
    max = - float('inf')
    iterationNumber = None      # Denotes on which iteration the max value was found
    for timeT in values:
        if values[timeT] > max:
            max = values[timeT]
            iterationNumber = timeT
    return (max, iterationNumber)

# Policy of the given state -- Calculated as max number in the Utility of each adjacent state where agent can travel to.
def policyOfUtility(stateS):
    nextStates = getNextStates(stateS)
    calculatedMaxState = (None, - float('inf'), None)       # Contains tuple (state, maxValue, iterationOfMaxValue)
    for nextState in nextStates:
        dummy = getMaxValueFromState(nextState)
        if calculatedMaxState[1] < dummy[0]:
            calculatedMaxState = (nextState, dummy[0], dummy[1])
    print('calculatedMaxState : ',calculatedMaxState)
    return calculatedMaxState[0]

# To define agent movement:
def agentUsingUtility():
    tiebreaker = 0
    statesTaken = [(gVar.agentPos, gVar.prey.getCurrNode(), gVar.predator.getCurrNode())]
    prediction = [(gVar.agentPos, gVar.prey.getCurrNode(), gVar.predator.getCurrNode())]
    while tiebreaker < 5000:
        currAgentPos = gVar.agentPos
        currPredatorPos = gVar.predator.getCurrNode()
        currPreyPos = gVar.prey.getCurrNode()
        if currAgentPos == currPredatorPos and currAgentPos == currPreyPos:
            # return 2
            return (2, statesTaken, prediction)
        elif currAgentPos == currPredatorPos:
            # return -1
            return (1, statesTaken, prediction)
        elif currAgentPos == currPreyPos:
            # return 1
            return (0, statesTaken, prediction)
        getToState = policyOfUtility((currAgentPos, currPreyPos, currPredatorPos))
        # Checks to see if getToState is correct.        
        nextAgentPositions = gVar.g.getNextNodes(currAgentPos)
        nextAgentPositions.append(currAgentPos)
        nextPreyPositions = gVar.g.getNextNodes(currPreyPos)
        nextPreyPositions.append(currPreyPos)
        nextPredatorPositions = gVar.g.getNextNodes(currPredatorPos)
        nextPredatorPositions.append(currPredatorPos)
        if getToState[0] not in nextAgentPositions:
            return 3
        if getToState[1] not in nextPreyPositions:
            return 4
        if getToState[2] not in nextPredatorPositions:
            return 5
        # Allocating Agent's next best position as received from policy function of the Utility State:
        gVar.agentPos = getToState[0]
        preyPredatorMovement()
        prediction.append(getToState)
        statesTaken.append(((gVar.agentPos, gVar.prey.getCurrNode(), gVar.predator.getCurrNode())))
        if currAgentPos == currPredatorPos and currAgentPos == currPreyPos:
            # return 2
            return (2, statesTaken, prediction)
        elif currAgentPos == currPredatorPos:
            # return -1
            return (1, statesTaken, prediction)
        elif currAgentPos == currPreyPos:
            # return 1
            return (0, statesTaken, prediction)

        tiebreaker += 1
        # gVar.pre = getToState[0]
        # currPreyPos = gVar.prey.getCurrNode()
    return (-2, statesTaken, prediction)


# To perform different agents on loop
def agentUsingUtilityLoop():
    result = []
    for i in range(100):
        print('Agent run: ',i+1)
        gVar.agentPos = placeEntities(gVar.g)
        showEntityPositions()
        result.append(agentUsingUtility())
        showEntityPositions()
    # print('Customized Agent Result = ' + str(result))
    # countTotal(result)
    # writeToCSV(result, gVar.agentNo)
    return result


if __name__=='__main__':
    gVar.g = gVar.Graph(gVar.size)
    start_time = datetime.datetime.now()

    initializeStates()
    print(gVar.states)
    print(len(gVar.states))

    create_env()
    # stateToCheck = random.choice(gVar.states)
    # neighborsOfState = getNextStates(stateToCheck)
    # print('neighborsOfState : ',neighborsOfState)
    # print('Count neighborsOfState : ',len(neighborsOfState))

    calculateUtility()
    result = agentUsingUtilityLoop()
    writeToCSV(gVar.utilityOfStates)
    print('Result = ', result)

    end_time = datetime.datetime.now()
    print('Start time : '+str(start_time))
    print('End time : '+str(end_time))
    print('Total time : '+str(end_time-start_time))
