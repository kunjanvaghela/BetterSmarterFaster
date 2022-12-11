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
import pickle



# To initiate different state spaces for the graph
def initializeStates():
    # States represent unique state possible in the environment: (Agent, Prey, Predator) positions
    for i in range(gVar.size):
        for j in range(gVar.size):
            for k in range(gVar.size):
                gVar.states.append((i,j,k))
                gVar.utilityOfNextAction[(i,j,k)] = {}

# To initialize reward of being in the current state
def initializeRewardOfStateS():
    # Initializes reward of being in a particular state
    for s in gVar.states:
        gVar.utilityAtTimeTMinus1[s] = {}
        # reward = getRewardOfState(s)
        # gVar.utilityAtTimeTMinus1[s] = reward
        gVar.utilityAtTimeTMinus1[s] = 0.     # reward


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

# To set probability matrix/relationship for the current graph along with all the states
def calculateProbability():
    for state in gVar.states:

        if state[0] == state[1] or state[0] == state[2]:
            gVar.probabilityStateTransition[state] = dict()
            gVar.probabilityStateTransition[state][state] = 1.
            continue

        # Get probability of all the next cells of the predator
        nextPredatorPos = gVar.g.getNextNodes(state[2])
        probPredator = {i:float(0) for i in nextPredatorPos}
        bfsResult = gVar.g.breadthFirstSearch(state[2], state[0])[0]  # Gets the path from Predator to Agent positions

        #For each pred pos, go for BFS result
        minBfsLen = 10000
        minDistPredPos = []
        for nextPred in nextPredatorPos:
            bfsLength = gVar.g.breadthFirstSearch(nextPred, state[0])[1]  # Gets the path from Predator to Agent positions
            if minBfsLen >= bfsLength:
                minBfsLen = bfsLength
                minDistPredPos.append(nextPred)

        if len(bfsResult) == 1:
            # If Predator and Agent Positions are the same, then bfs path will be 1 only. Adding current node in nextPredatorPos list.
            probPredator[state[2]] = float(0)
            nextPredatorPos.append(state[2])
        for nextPred in nextPredatorPos:
            if len(bfsResult) > 1:
                # Adding 0.6 probability of predator going to the next node lying in BFS path to Agent
                # if nextPred == bfsResult[1]:
                if nextPred in minDistPredPos:
                    probPredator[nextPred] += 0.6 / len(minDistPredPos)
            elif len(bfsResult) == 1:
                # Since Predator will be in same cell only in this condition, adding 0.6 probability to the current cell of the predator.
                if nextPred == bfsResult[0]:
                    probPredator[nextPred] += 0.6
            # Divinding 0.4 probability into all other nodes that predator can go to
            probPredator[nextPred] += float((1 - 0.6) / len(nextPredatorPos))
        
        # Calculate probability of Prey going to all the next cells from the current Prey Position
        nextPreyPos = gVar.g.getNextNodes(state[1])
        nextPreyPos.append(state[1])
        probPrey = {i:float(0) for i in nextPreyPos}
        for nextPrey in nextPreyPos:
            probPrey[nextPrey] += (1. / len(nextPreyPos))

        # Calculate timestamp t's Utility
        nextStates = getNextStates(state)
        gVar.probabilityStateTransition[state] = dict()
        for nextState in nextStates:
            if nextState[1] in probPrey and nextState[2] in probPredator:
                gVar.probabilityStateTransition[state][nextState] = (probPrey[nextState[1]] * probPredator[nextState[2]])
            else:
                gVar.probabilityStateTransition[state][nextState] = 0.0

    for state in gVar.states:
        nextStates = gVar.probabilityStateTransition[state]
        sumOfNextStates = sum(nextStates.values())
        if state[0] == state[1] or state[0] == state[2]:
            neighbourCount = 1
        else:
            neighbourCount = len(gVar.g.getNextNodes(state[0])) + 1
        if abs(sumOfNextStates-neighbourCount) > 0.0001:
            print("Error")
            # print(state)
            # print(nextStates)
            print(sumOfNextStates)
            print(neighbourCount)
            quit()

def getUniqueActions(stateSpace):
    actions = set()
    for s in stateSpace:
        actions.add(s[0])
    return actions

def getRewardOfState(stateS):
    reward = gVar.rewardNormal
    if stateS[0] == stateS[2]:
        reward = gVar.rewardPredator      # Reward of being with the predator
    elif stateS[0] == stateS[1]:
        reward = gVar.rewardPrey           # Reward of being with the prey
    return reward

def calculateUtility(prob, state):
    # return (prob * gVar.utilityAtTimeTMinus1[state])
    # val = -1 + gVar.utilityAtTimeTMinus1[state]  #jn
    val = gVar.utilityAtTimeTMinus1[state]
    # print("utility ", val)
    if np.isnan(val):
        val = 0
    # return (prob * (getRewardOfState(state) + gVar.utilityAtTimeTMinus1[state]))
    # print("probval", prob, val)
    if prob > 0.0:
        return (prob * val)
    return 0.0

def calculateOptimalUtility():
    initializeRewardOfStateS()
    statesToCheck = gVar.states
    i = 1
    while True:

        if i > 1:
            gVar.utilityAtTimeTMinus1 = gVar.utilityAtTimeT.copy()

        for state in statesToCheck:
            if state[0] == state[2]:
                gVar.utilityAtTimeT[state] = gVar.utilityAtTimeTMinus1[state]
                gVar.utilityOfNextAction[state] = {state[0] : gVar.rewardPredator}
                gVar.utilityAtTimeT[state] = gVar.rewardPredator
                # gVar.utilityAtTimeTMinus1[state] = gVar.rewardPredator
            elif state[0] == state[1]:
                gVar.utilityAtTimeT[state] = gVar.utilityAtTimeTMinus1[state]
                gVar.utilityOfNextAction[state] = {state[0] : gVar.rewardPrey}
                gVar.utilityAtTimeT[state] = gVar.rewardPrey
                # gVar.utilityAtTimeTMinus1[state] = gVar.rewardPrey
            elif state[2] in gVar.g.getNextNodes(state[0]):
                gVar.utilityAtTimeT[state] = gVar.utilityAtTimeTMinus1[state]
                gVar.utilityOfNextAction[state] = {state[0] : gVar.rewardPredator}
                gVar.utilityAtTimeT[state] = gVar.rewardPredator
                # gVar.utilityAtTimeTMinus1[state] = gVar.rewardPrey
            else:
                # Get transition probability of all current cells
                transitionProbsCurrentState = gVar.probabilityStateTransition[state]
                # Get all the possible next states of the agent
                nextActionPositions = getUniqueActions(transitionProbsCurrentState)
                gVar.utilityOfNextAction[state] = {nextAction: 0. for nextAction in nextActionPositions}
                # Calculating optimal utility
                for nextState in transitionProbsCurrentState.keys():
                    # print("so far ", gVar.utilityOfNextAction[state][nextState[0]])
                    if (np.isnan(gVar.utilityOfNextAction[state][nextState[0]])):
                        print("It's nan ")
                        quit()
                    if (np.isnan(getRewardOfState(nextState) + calculateUtility(transitionProbsCurrentState[nextState], nextState))):
                        print("Check {0} {1} {2}", gVar.utilityOfNextAction[state][nextState[0]] , calculateUtility(transitionProbsCurrentState[nextState], nextState), getRewardOfState(nextState))
                        quit()
                    # gVar.utilityOfNextAction[state][nextState[0]] += getRewardOfState(nextState) + calculateUtility(transitionProbsCurrentState[nextState], nextState)    #jn
                    gVar.utilityOfNextAction[state][nextState[0]] += calculateUtility(transitionProbsCurrentState[nextState], nextState)
                # Store optimal utility of time t in utilityAtTimeT
                # gVar.utilityAtTimeT[state] = max(gVar.utilityOfNextAction[state].values())
                gVar.utilityAtTimeT[state] = -1 + max(gVar.utilityOfNextAction[state].values())
        
        maxi = - float('inf')
        for st in gVar.utilityAtTimeT:
            diff = abs(gVar.utilityAtTimeT[st] - gVar.utilityAtTimeTMinus1[st])
            if diff > maxi:
                maxi = diff
        err = 0.000001
        print('Calculation for i = ',i,'; Max Error = ',maxi)
        if maxi < err:
            break

        i += 1


# To get max value from the utility function                --- Can remove
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
    # nextStates = getNextStates(stateS)
    # calculatedMaxState = (None, - float('inf'), None)       # Contains tuple (state, maxValue, iterationOfMaxValue)
    # for nextState in nextStates:
    #     dummy = getMaxValueFromState(nextState)
    #     if calculatedMaxState[1] < dummy[0]:
    #         calculatedMaxState = (nextState, dummy[0], dummy[1])
    # print('calculatedMaxState : ',calculatedMaxState)
    # return calculatedMaxState[0]

    # Get transition probability of all current cells
    # transitionProbsCurrentState = gVar.probabilityStateTransition[stateS]
    # # Get all the possible next states of the agent
    # nextActionPositions = getUniqueActions(transitionProbsCurrentState)
    # utilityOfNextAction = {nextAction: 0. for nextAction in nextActionPositions}
    # # Calculating optimal utility
    # for nextState in transitionProbsCurrentState.keys():
    #     utilityOfNextAction[nextState[0]] += calculateUtility(transitionProbsCurrentState[nextState], nextState)
    # Store optimal utility of time t in utilityAtTimeT
    maxi = max(gVar.utilityOfNextAction[stateS].values())
    for i in gVar.utilityOfNextAction[stateS]:
        if maxi == gVar.utilityOfNextAction[stateS][i]:
            return i
    

# To define agent movement:
def agentUsingUtility():
    tiebreaker = 0
    statesTaken = [(gVar.agentPos, gVar.prey.getCurrNode(), gVar.predator.getCurrNode())]
    prediction = [(gVar.agentPos, gVar.prey.getCurrNode(), gVar.predator.getCurrNode())]
    while tiebreaker < 10000:
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
        if getToState not in nextAgentPositions:
            return 3
        # if getToState not in nextPreyPositions:
        #     return 4
        # if getToState not in nextPredatorPositions:
        #     return 5
        # Allocating Agent's next best position as received from policy function of the Utility State:
        gVar.agentPos = getToState

        ## Add checkstate condition
        if currAgentPos == currPredatorPos and currAgentPos == currPreyPos:
            # return 2
            return (2, statesTaken, prediction)
        elif currAgentPos == currPredatorPos:
            # return -1
            return (1, statesTaken, prediction)
        elif currAgentPos == currPreyPos:
            # return 1
            return (0, statesTaken, prediction)

        #preyPredatorMovement()
        preyMovement()
        ## Add checkstate condition
        if currAgentPos == currPredatorPos and currAgentPos == currPreyPos:
            # return 2
            return (2, statesTaken, prediction)
        elif currAgentPos == currPredatorPos:
            # return -1
            return (1, statesTaken, prediction)
        elif currAgentPos == currPreyPos:
            # return 1
            return (0, statesTaken, prediction)

        predatorMovement()
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
        # print("t ", tiebreaker) 
        # print(showEntityPositions())
        # gVar.pre = getToState[0]
        # currPreyPos = gVar.prey.getCurrNode()
    return (-2, statesTaken, prediction)


# To perform different agents on loop
def agentUsingUtilityLoop():
    result = []
    for i in range(gVar.agentIterations):
        print('Agent run: ',i+1)
        gVar.agentPos = placeEntities(gVar.g)
        showEntityPositions()
        result.append(agentUsingUtility())
        showEntityPositions()
    # print('Customized Agent Result = ' + str(result))
    # countTotal(result)
    # writeToCSV(result, gVar.agentNo)
    return result

def countIt(result):
    count = 0
    res = {}
    for r in result:
        count += 1
        # print(r)
        if r[0] not in res.keys():
            res[r[0]] = 1
        else:
            res[r[0]] += 1
    print('Count = ',count)
    print('res : ',res)

#input data is dict.
# OutputCSV: state, agentPos, preyPos, predPos, distOfAgPrey, distOfAgPredator, Utility
def writeToCSVForModel(data):
    with open('Data4/UtilitiesFinal.csv', 'w', newline = '') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['State', 'AgentPos', 'PreyPos', 'PredPos', 'AgentPreyDist', 'AgentPredDist', 'Utility'])
        for i in data:
            dAgPy = gVar.g.breadthFirstSearch(i[0], i[1])[1]
            dAgPred = gVar.g.breadthFirstSearch(i[0], i[2])[1]
            writer.writerow([i, i[0], i[1], i[2], dAgPy, dAgPred, max(data[i].values())])
            # file.write(str(i) , str(data[i]))
        file.close()
        # print("Successfully written to file {}", 'a_' + str(agentNo) + '.csv')
    print("Successfully written features to csv file.")

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

    calculateProbability()
    writeToFile(gVar.probabilityStateTransition, 'Data4/ProbabilityStateTransition')
    calculateOptimalUtility()
    writeToFile(gVar.utilityAtTimeTMinus1, 'Data4/utilityAtTimeTMinus1')
    writeToFile(gVar.utilityAtTimeT, 'Data4/utilityAtTimeT')
    writeToFile(gVar.utilityOfNextAction, 'Data4/utilityOfNextAction')
    result = agentUsingUtilityLoop()
    # writeToCSV(gVar.utilityOfStates)
    print('Result = ', result)

    countIt(result)

    end_time = datetime.datetime.now()
    print('Start time : '+str(start_time))
    print('End time : '+str(end_time))
    print('Total time : '+str(end_time-start_time))

    writeToCSVForModel(gVar.utilityOfNextAction)

    mat = gVar.g.adjMatrix
    with open('Data4/graph4.pickle', 'wb') as f:
	    pickle.dump(mat, f)

    #Save features and U* value. => Saved in UtilitesFinal.csv