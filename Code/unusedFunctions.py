import random
import csv
import networkx as nx

from customizedFunctions import *




# Takes list of nextNodes, and returns lists nearestPredatorNode and farthestPredatorNode, each containing respective tuple (nextNode, distance)
def getNearestFarthestPredator(nextNodes, predatorNode = -1):
    nearestPredatorNode, farthestPredatorNode = None, None
    global g, prey, predator
    # If predatorNode value is not received from parent, then predator current node is known in the environment
    if predatorNode == -1:
        predatorNode = predator.getCurrNode()
    # Below for loop retrives path and distance to prey and predator of each node adjacent to agent's position node and stores in nextNodeAttributes dict
    nextNodeAttributes = {}
    for node in nextNodes:
        # Get Predator nodes
        predatorPathAndDistance = g.breadthFirstSearch(node, predatorNode)
        # Dictionary with {int node: tuple( tuple(prey bfs path, distance to prey), tuple(predator bfs path, distance to predator) )}
        # Stored tuple of tuples instead of list of tuples for better memory efficiency
        nextNodeAttributes[node] = predatorPathAndDistance

        # Checks to store nearest node and distance to the predator for agent
        if nearestPredatorNode is None:
            nearestPredatorNode = [(node, predatorPathAndDistance[1])]
        elif nearestPredatorNode[0][1] > predatorPathAndDistance[1]:     # Will replace the nearestPredatorNode if the next node's distance is less from Predator
            nearestPredatorNode = [(node, predatorPathAndDistance[1])]
        elif nearestPredatorNode[0][1] == predatorPathAndDistance[1]:     # Will append one more nearestPredatorNode if another node's distance is equal to previously discovered node
            nearestPredatorNode.append((node, predatorPathAndDistance[1]))

        # Checks to store farthest node and distance to the predator for agent
        if farthestPredatorNode is None:
            farthestPredatorNode = [(node, predatorPathAndDistance[1])]
        elif farthestPredatorNode[0][1] < predatorPathAndDistance[1]:     # Will replace the farthestPredatorNode if the next node's distance is more towards Predator
            farthestPredatorNode = [(node, predatorPathAndDistance[1])]
        elif farthestPredatorNode[0][1] == predatorPathAndDistance[1]:     # Will append one more farthestPredatorNode if another node's distance is equal to previously discovered node
                farthestPredatorNode.append((node, predatorPathAndDistance[1]))
    return nearestPredatorNode, farthestPredatorNode, nextNodeAttributes

# Takes list of nextNodes, and returns lists nearestPredatorNode, nearestPreyNode, farthestPredatorNode, farthestPreyNode, each containing respective tuple (nextNode, distance)
def getNearestFarthestPreyPredator(nextNodes, preyNode = -1, predatorNode = -1):
    nearestPredatorNode, nearestPreyNode, farthestPredatorNode, farthestPreyNode = None, None, None, None
    global g, prey, predator
    # If preyNode/predatorNode value is not received from parent, assumed that the predator/prey current node is known in the environment 
    if preyNode == -1:
        preyNode = prey.getCurrNode()
    if predatorNode == -1:
        predatorNode = predator.getCurrNode()
    # Below for loop retrives path and distance to prey and predator of each node adjacent to agent's position node and stores in nextNodeAttributes dict
    nextNodeAttributes = {}
    for node in nextNodes:
        # Get Prey and Predator nodes
        preyPathAndDistance = g.breadthFirstSearch(node, preyNode)
        predatorPathAndDistance = g.breadthFirstSearch(node, predatorNode)
        # Dictionary with {int node: tuple( tuple(prey bfs path, distance to prey), tuple(predator bfs path, distance to predator) )}
        # Stored tuple of tuples instead of list of tuples for better memory efficiency
        nextNodeAttributes[node] = (preyPathAndDistance, predatorPathAndDistance)

        # Checks to store nearest node and distance to the predator for agent
        if nearestPredatorNode is None:
            nearestPredatorNode = [(node, predatorPathAndDistance[1])]
        elif nearestPredatorNode[0][1] > predatorPathAndDistance[1]:     # Will replace the nearestPredatorNode if the next node's distance is less from Predator
            nearestPredatorNode = [(node, predatorPathAndDistance[1])]
        elif nearestPredatorNode[0][1] == predatorPathAndDistance[1]:     # Will append one more nearestPredatorNode if another node's distance is equal to previously discovered node
            nearestPredatorNode.append((node, predatorPathAndDistance[1]))

        # Checks to store nearest node and distance to the prey for agent
        if nearestPreyNode is None:
            nearestPreyNode = [(node, preyPathAndDistance[1])]     # List of tuple of nearestPreyNode
        elif nearestPreyNode[0][1] > preyPathAndDistance[1]:     # Will replace the nearestPreyNode if the next node's distance is less from Prey
            # print('------In If--------')
            # print('Current nearestPreyNode : '+str(nearestPreyNode))
            # print('This node preyPathAndDistance : '+str(preyPathAndDistance))
            nearestPreyNode = [(node, preyPathAndDistance[1])]
            # print('Current nearestPreyNode : '+str(nearestPreyNode))
        elif nearestPreyNode[0][1] == preyPathAndDistance[1]:     # Will append another nearestPreyNode if the next node's distance is also nearest
            # print('------In elif--------')
            # print('Current nearestPreyNode : '+str(nearestPreyNode))
            # print('This node preyPathAndDistance : '+str(preyPathAndDistance))
            nearestPreyNode.append((node, preyPathAndDistance[1]))
            # print('Current nearestPreyNode : '+str(nearestPreyNode))
        else:
            pass
            # print('------In else--------')
            # print('Current nearestPreyNode : '+str(nearestPreyNode))
            # print('This node preyPathAndDistance : '+str(preyPathAndDistance))

        # Checks to store farthest node and distance to the predator for agent
        if farthestPredatorNode is None:
            farthestPredatorNode = [(node, predatorPathAndDistance[1])]
        elif farthestPredatorNode[0][1] < predatorPathAndDistance[1]:     # Will replace the farthestPredatorNode if the next node's distance is more towards Predator
            farthestPredatorNode = [(node, predatorPathAndDistance[1])]
        elif farthestPredatorNode[0][1] == predatorPathAndDistance[1]:     # Will append one more farthestPredatorNode if another node's distance is equal to previously discovered node
                farthestPredatorNode.append((node, predatorPathAndDistance[1]))

        # Checks to store farthest node and distance to the prey for agent
        if farthestPreyNode is None:
            farthestPreyNode = [(node, preyPathAndDistance[1])]     # List of tuple of farthestPreyNode
        elif farthestPreyNode[0][1] < preyPathAndDistance[1]:     # Will replace the farthestPreyNode only if the next node's distance is more towards Prey
            farthestPreyNode = [(node, preyPathAndDistance[1])]
        elif farthestPreyNode[0][1] == preyPathAndDistance[1]:     # Will append one more farthestPreyNode if another node's distance is equal to previously discovered node
            farthestPreyNode.append((node, preyPathAndDistance[1]))
    return nearestPredatorNode, nearestPreyNode, farthestPredatorNode, farthestPreyNode, nextNodeAttributes



def agentOneDefinedSteps(nextNodeAttributes, nearestPreyNode, nearestPredatorNode):
    global agentPos
    # Distance from Agent: x = dist of Predator from Agent; y = dist of Prey from Agent
    x = nearestPredatorNode[0][1]+1
    y = nearestPreyNode[0][1]+1

    # nextNodeAttributes is a dictionary: {[node] = (preyPathAndDistance, predatorPathAndDistance)}
    neighborForCondition1, neighborForCondition2, neighborForCondition3, neighborForCondition4, neighborForCondition5, neighborForCondition6 = [], [], [], [], [], []
    for i in nextNodeAttributes:
        # print(nextNodeAttributes[i][0][1])
        # To get neighbors that satisfy condition 1: Neighbors that are closer to the Prey and farther from the Predator.
        if nextNodeAttributes[i][0][1] < y and nextNodeAttributes[i][1][1] > x:
            neighborForCondition1.append(i)
        # To get neighbors that satisfy condition 2: Neighbors that are closer to the Prey and not closer to the Predator.
        if nextNodeAttributes[i][0][1] < y and nextNodeAttributes[i][1][1] == x:
            neighborForCondition2.append(i)
        # To get neighbors that satisfy condition 3: Neighbors that are not farther from the Prey and farther from the Predator.
        if nextNodeAttributes[i][0][1] <= y and nextNodeAttributes[i][1][1] > x:
            neighborForCondition3.append(i)
        # To get neighbors that satisfy condition 4: Neighbors that are not farther from the Prey and not closer to the Predator.
        if nextNodeAttributes[i][0][1] <= y and nextNodeAttributes[i][1][1] >= x:
            neighborForCondition4.append(i)
        # To get neighbors that satisfy condition 5: Neighbors that are farther from the Predator.
        if nextNodeAttributes[i][1][1] > x:
            neighborForCondition5.append(i)
        # To get neighbors that satisfy condition 6: Neighbors that are not closer to the Predator.
        if nextNodeAttributes[i][1][1] == x:
            neighborForCondition6.append(i)
    
    # print('neighborForCondition1 : ' + str(neighborForCondition1)) 
    # print('neighborForCondition2 : ' + str(neighborForCondition2))  
    # print('neighborForCondition3 : ' + str(neighborForCondition3))
    # print('neighborForCondition4 : ' + str(neighborForCondition4))
    # print('neighborForCondition5 : ' + str(neighborForCondition5))
    # print('neighborForCondition6 : ' + str(neighborForCondition6))

    # Set of rules based on Agent 1's Logic:
    nextAgentPos = agentPos
    # Neighbors that are closer to the Prey and farther from the Predator.
    if neighborForCondition1 != []:
        nextAgentPos = random.choice(neighborForCondition1)
    # Neighbors that are closer to the Prey and not closer to the Predator.
    elif neighborForCondition2 != []:
        nextAgentPos = random.choice(neighborForCondition2)
    # Neighbors that are not farther from the Prey and farther from the Predator.
    elif neighborForCondition3 != []:
        nextAgentPos = random.choice(neighborForCondition3)
    # Neighbors that are not farther from the Prey and not closer to the Predator.
    elif neighborForCondition4 != []:
        nextAgentPos = random.choice(neighborForCondition4)
    # Neighbors that are farther from the Predator.
    elif neighborForCondition5 != []:
        nextAgentPos = random.choice(neighborForCondition5)
    # Neighbors that are not closer to the Predator.
    elif neighborForCondition6 != []:
        nextAgentPos = random.choice(neighborForCondition6)
    # Sit still and pray.
    else:
        print('Something')
        pass
    return nextAgentPos



def getHeuristic(prob, distance):
    if distance<1:
        # print('Error')
        return (prob)
    return (prob/distance)

def agentCustomizedDefinedSteps(predatorPositionKnown = 0, preyMidStateBelief = preyStateBelief, predatorMidStateBelief = predatorStateBelief, agentFaultyDroneUpdate = 0):
    global agentPos, g, pointSystem, surveyedNode
    probOfFaultyDroneUpdateAgent = random.random()
    # Get the nextAgentPos if predator is not in nearest path of the possible nodes

    if agentFaultyDroneUpdate == 1 and probOfFaultyDroneUpdateAgent > 0.9 and surveyedNode != -1:
        predatorMaxPositionProbability = surveyedNode
    else:
        # Get the cells with maximum probability of having Predator
        predatorMaxPositionProbability = findMaxProbability(predatorMidStateBelief, nextStep=1)        # To check against Predator Nodes
        # print('predatorMaxPositionProbability: ',predatorMaxPositionProbability,'; predatorMidStateBelief[predatorMaxPositionProbability[0]] : ',predatorMidStateBelief[predatorMaxPositionProbability[0]])
        # print('predatorMidStateBelief : ',predatorMidStateBelief)

        # Logic for this if - If maxProbability of a node is > 0.5, then there will be only 1 node with higher probability
        # And this cell will be used to pre-check in below logic if the predator is nearby
        if predatorMidStateBelief[predatorMaxPositionProbability[0]] < 0.51:
            predatorMaxPositionProbability = -1
        else:
            predatorMaxPositionProbability = predatorMaxPositionProbability[0]
    # print('predatorMaxPositionProbability : ',predatorMaxPositionProbability)
    # If there is a cell with higher Predator probability (>= 0.51)
    if predatorMaxPositionProbability != -1:
        pathToPredator = g.breadthFirstSearch(agentPos, predatorMaxPositionProbability)
        # print('pathToPredator : ',pathToPredator)
        # if pathToPredator[1] == 0:
        #     print('Check condition.')
        # Check if path to that predator node is < 5
        if pathToPredator[1] < 5:
            # If predator is nearby, run from that high probability predator cell
            # Checks based on agentFaultyDroneUpdate (Agent 9) logic
            if agentFaultyDroneUpdate == 1 and probOfFaultyDroneUpdateAgent > 0.9 and surveyedNode != -1:
                currPredMaxProb = surveyedNode
            else:
                currPredMaxProb = random.choice(findMaxProbability(predatorStateBelief))
            if predatorPositionKnown == 0:
                # If Predator Position is not known for the agent being checked, then pass the predator position as received from Probability Belief of the next step.
                nearestPredatorNode, farthestPredatorNode, nextNodeAttributes = getNearestFarthestPredator(g.getNextNodes(agentPos), predatorNode=currPredMaxProb)  # predatorNode=predatorMaxPositionProbability
            else:   # With below else: Total - 0 = 7701, 1 = 2299, -1 = 0
                # If Predator Position is known for the agent being checked, then pass the predator position will be enriched in the getNearestFarthestPredator definition.
                nearestPredatorNode, farthestPredatorNode, nextNodeAttributes = getNearestFarthestPredator(g.getNextNodes(agentPos), predatorNode=-1)
            # print('farthestPredatorNode: ',farthestPredatorNode,'; nearestPredatorNode: ',nearestPredatorNode)
            # For the adjacent nodes of the Agent, check the farthest and nearest nodes from the predator.
            for farthestnode in farthestPredatorNode:
                # If farthest node and nearest node are different, then the next cell will take us farther from the Predator.
                if farthestnode[1] != nearestPredatorNode[0][1]:
                    return farthestnode[0]
                else:
                    # return agentPos
                    # Without this: Total - 0 = 7067, 1 = 2933, -1 = 0
                    # With this: Total - 0 = 7494, 1 = 2506, -1 = 0
                    # If distance from the farthest node and the nearest node are same, then can go to the node only if Predator is away from 2 cells.
                    # This can give Agent another adjacent node option which can take Agent away from the Predator. 
                    if nearestPredatorNode[0][1] > 2:
                        return farthestnode[0]
                    # If the distance is close, Agent stays and hopes for the best.
                    return agentPos


    ## Calculate Points for this move
    pointSystem = {}
    for i in range(size):
        pointSystem[i] = float(0)
    # Get Possible cell locations of the Prey and Predator from belief system and set Point System.
    preyPositionsProbability = getNonZeroProbabilityIndices(preyMidStateBelief)     #findMaxProbability(preyMidStateBelief)
    predatorPositionProbability = getNonZeroProbabilityIndices(predatorMidStateBelief)  #findMaxProbability(predatorMidStateBelief)
    for n in preyPositionsProbability:
        pointSystem[n] += (getHeuristic(preyMidStateBelief[n], g.breadthFirstSearch(agentPos, n)[1]))
    for n in predatorPositionProbability:
        pointSystem[n] -= getHeuristic(predatorMidStateBelief[n], g.breadthFirstSearch(agentPos, n)[1])
    
    pointSystem_placeholder = pointSystem.copy()
    while len(pointSystem_placeholder) != 0:
        currMaxPointNodes = findMaxProbability(pointSystem_placeholder, nextStep=1)     # List received
        if pointSystem[currMaxPointNodes[0]] < 0.0001:
            break
        while len(currMaxPointNodes) != 0:
            n = random.choice(currMaxPointNodes)
            pathAgentToN = g.breadthFirstSearch(agentPos, n)[0]
            if len(pathAgentToN) == 1:
                currMaxPointNodes.remove(n)
                pointSystem_placeholder.pop(n)
                continue
            if predatorMaxPositionProbability != -1:
                if predatorMaxPositionProbability in pathAgentToN:
                    if pathAgentToN.index(predatorMaxPositionProbability) < 3 and pathAgentToN.index(predatorMaxPositionProbability) > 0:
                        currMaxPointNodes.remove(n)
                        pointSystem_placeholder.pop(n)
                        continue
            return pathAgentToN[1]
    print('Check this condition')
    # Get Farthest Node from Predator Node determined, and run!!!!!!!
    nearestPredatorNode, nearestPreyNode, farthestPredatorNode, farthestPreyNode, nextNodeAttributes = getNearestFarthestPreyPredator(g.getNextNodes(agentPos), predatorNode=predatorMaxPositionProbability)
    return random.choice(farthestPredatorNode)[0]

def surveyLogic(preyPositionKnown = 0, predatorPositionKnown = 0, defectiveDrone = 0, beliefForFaultyDrone = 0):
    global predatorFoundCounter, preyFoundCounter
    if preyPositionKnown == 0 and predatorPositionKnown == 0:
        predatorPosKnown = isPredatorPosKnown()
        if predatorPosKnown:
            surveyedNode, preyPresent = surveyNode(0, defectiveDrone)
            if preyPresent == 1:
                preyFoundCounter += 1
            updateBelief(0, surveyedNode, preyPresent, beliefForFaultyDrone)
        else:
            surveyedNode, predatorPresent = surveyNode(1, defectiveDrone)
            if predatorPresent == 1:
                predatorFoundCounter += 1
            updateBelief(1, surveyedNode, predatorPresent, beliefForFaultyDrone)          # preyPresent = 1 if Prey present else 0
            if surveyedNode == prey.getCurrNode():
                updateBelief(0, surveyedNode, 1)
            else:
                updateBelief(0, surveyedNode, 0)
    elif preyPositionKnown == 0 and predatorPositionKnown == 1:
        surveyedNode, preyPresent = surveyNode(0, defectiveDrone)
        if preyPresent == 1:
            preyFoundCounter += 1
        updateBelief(0, surveyedNode, preyPresent, beliefForFaultyDrone)
    elif preyPositionKnown == 1 and predatorPositionKnown == 0:
        surveyedNode, predatorPresent = surveyNode(1, defectiveDrone)
        if predatorPresent == 1:
            predatorFoundCounter += 1
        updateBelief(1, surveyedNode, predatorPresent, beliefForFaultyDrone)


def surveyOrMoveAgentMovement(preyPositionKnown, predatorPositionKnown, easilyDistractedPredator, defectiveDrone, beliefForFaultyDrone = 0):
    global g, predator, prey, agentPos, preyStateBelief, predatorStateBelief, preyFoundCounter, predatorFoundCounter
    threshold = 0.1
    preyMidStateBelief, predatorMidStateBelief = updateBeliefNextStep(easilyDistractedPredator, nextStepLookup= 1)

    # Get the cells with maximum probability of having Predator
    # predatorMaxPositionProbability = findMaxProbability(predatorMidStateBelief, nextStep=1)        # To check against Predator Nodes
    predatorMaxPositionProbability = getDistanceWithClosestPredatorProximity(predatorStateBelief = predatorMidStateBelief)

    # Logic for this if - If maxProbability of a node is > 0.5, then there will be only 1 node with higher probability
    # And this cell will be used to pre-check in below logic if the predator is nearby
    if predatorMidStateBelief[predatorMaxPositionProbability[0]] < threshold:
        predatorMaxPositionProbability = -1
    else:
        predatorMaxPositionProbability = predatorMaxPositionProbability[0]
    # If there is a cell with higher Predator probability (>= 0.51)     # Agent Moves if Predator is clove by
    if predatorMaxPositionProbability != -1:
        pathToPredator = g.breadthFirstSearch(agentPos, predatorMaxPositionProbability)
        # Check if path to that predator node is < 5
        if pathToPredator[1] < 5:
            # If predator is nearby, run from that high probability predator cell
            if predatorPositionKnown == 0:
                # If Predator Position is not known for the agent being checked, then pass the predator position as received from Probability Belief of the next step.
                nearestPredatorNode, farthestPredatorNode, nextNodeAttributes = getNearestFarthestPredator(g.getNextNodes(agentPos), predatorNode=random.choice(findMaxProbability(predatorStateBelief)))  # predatorNode=predatorMaxPositionProbability
            else:
                # If Predator Position is known for the agent being checked, then pass the predator position will be enriched in the getNearestFarthestPredator definition.
                nearestPredatorNode, farthestPredatorNode, nextNodeAttributes = getNearestFarthestPredator(g.getNextNodes(agentPos), predatorNode=-1)
            # For the adjacent nodes of the Agent, check the farthest and nearest nodes from the predator.
            for farthestnode in farthestPredatorNode:
                # If farthest node and nearest node are different, then the next cell will take us farther from the Predator.
                if farthestnode[1] != nearestPredatorNode[0][1]:
                    agentPos = farthestnode[0]
                else:
                    # If distance from the farthest node and the nearest node are same, then can go to the node only if Predator is away from 2 cells.
                    # This can give Agent another adjacent node option which can take Agent away from the Predator.
                    if nearestPredatorNode[0][1] > 2:
                        agentPos = farthestnode[0]
                    # If the distance is close, Agent stays and hopes for the best.
                    agentPos = agentPos     # Can Survey??
                    surveyLogic(preyPositionKnown, predatorPositionKnown, defectiveDrone, beliefForFaultyDrone)
    else:
        # Get the cells with maximum probability of having Prey
        preyMaxPositionProbability = findMaxProbability(preyMidStateBelief, nextStep=1)        # To check against Predator Nodes
        if preyMidStateBelief[preyMaxPositionProbability[0]] > threshold:
            # Move towards Prey
            closestPreyNodes = getDistanceWithClosestPredatorProximity(preyMidStateBelief)
            pathToPrey = g.breadthFirstSearch(agentPos, random.choice(closestPreyNodes))
            if pathToPrey[1] == 0:
                # Agent stays at same place as there is a chance that Prey will end up in the Agent's cells
                agentPos = pathToPrey[0][0]
                # Simultaneously surveys for the predator node
                surveyLogic(preyPositionKnown, predatorPositionKnown, defectiveDrone, beliefForFaultyDrone)
            else:
                agentPos = pathToPrey[0][1]
        else:
            surveyLogic(preyPositionKnown, predatorPositionKnown, defectiveDrone, beliefForFaultyDrone)

    # If Agent's next position is changing, need to update probability accordingly so that probability summation remains 1
    if agentPos != predator.getCurrNode():
        updateBelief(1, agentPos, 0)
    if agentPos != prey.getCurrNode():
        updateBelief(0, agentPos, 0)



def updateBeliefNextStep(distractedPredator = 0, nextStepLookup = 0):
    global preyStateBelief, predatorStateBelief
    if nextStepLookup == 0:
        updateBeliefOnEntityMovement(0)
        if distractedPredator==0:
            updateBeliefOnEntityMovement(1, distractedPredator=0)
        else:
            updateBeliefOnEntityMovement(1, distractedPredator=1)
    else:
        preyBelief = updateBeliefOnEntityMovement(0, nextStepLookup= nextStepLookup)
        if distractedPredator==0:
            predatorBelief = updateBeliefOnEntityMovement(1, distractedPredator=0, nextStepLookup= nextStepLookup)
        else:
            predatorBelief =updateBeliefOnEntityMovement(1, distractedPredator=1, nextStepLookup= nextStepLookup)
        return preyBelief, predatorBelief


# This function will return the max predator probability belief node which is nearest to the agent among other max predator probability belief nodes
def getDistanceWithClosestPredatorProximity(predatorStateBelief = predatorStateBelief):
    global agentPos
    pos = findMaxProbability(predatorStateBelief)
    minDist = -1
    minDistList = []
    for p in pos:
        if p == agentPos:
            continue
        pathReceived, distance = g.breadthFirstSearch(agentPos, p)
        if minDist == -1:
            minDist = distance
            minDistList.append(p)
        elif minDist == distance:
            minDistList.append(p)
        elif minDist > distance:
            minDist = distance
            minDistList = [p]
    return minDistList

# Argument: 0 - Survey for Prey, 1/others - Survey for Predator    -> Returns Tuple (randomly selected node with highest probability, 1 if Prey/Predator present else 0)
def surveyNode(surveyFor = 0, defectiveDrone = 0):
    global g, preyStateBelief, predatorStateBelief
    choiceDefective = random.random()
    if surveyFor == 0:
        pos = findMaxProbability(preyStateBelief)
        positionToCheck = random.choice(pos)
        if defectiveDrone==1 and choiceDefective < 0.1:
            return (positionToCheck, 0)
        if positionToCheck == prey.getCurrNode():
            return (positionToCheck, 1)
        else:
            return (positionToCheck, 0)
    else:
        minDistList = getDistanceWithClosestPredatorProximity()
        positionToCheck = random.choice(minDistList)
        if defectiveDrone==1 and choiceDefective < 0.1:
            return (positionToCheck, 0)
        if positionToCheck == predator.getCurrNode():
            return (positionToCheck, 1)
        else:
            return (positionToCheck, 0)

# Distributes probability from the input cell into other cells with non-zero probability values
# x : 0 = Prey Update, 1 = Predator Update
def distributeProbabilityFromCell(x, node, beliefForFaultyDrone = 0):
    # print('In distributeProbabilityFromCell with node : '+str(node))
    global g, preyStateBelief, predatorStateBelief
    if x == 0:
        checkStateBelief(0)
        nodesToAddP = [i for i in preyStateBelief if preyStateBelief[i] > 0]
    else:
        checkStateBelief(1)
        latestCurrentBelief = getStateBelief(1)
        nodesToAddP = [i for i in predatorStateBelief if predatorStateBelief[i] > 0]
    # print('nodesToAddP : '+str(nodesToAddP))
    if node in nodesToAddP:
        nodesToAddP.remove(node)
    for n in nodesToAddP:
        if beliefForFaultyDrone == 0:
            if x == 0:
                # print('n : '+ str(n))
                # print('node : '+ str(node))
                # print('preyStateBelief[node] : '+ str(preyStateBelief[node]))
                # print('len(nodesToAddP) : '+ str(len(nodesToAddP)))
                # preyStateBelief[n] += (preyStateBelief[node]/len(nodesToAddP))
                t = 1-preyStateBelief[node]
                # preyStateBelief[n] += (preyStateBelief[n]*t)
                preyStateBelief[n] = (preyStateBelief[n]/t)
                # checkStateBelief(0)
            else:
                # predatorStateBelief[n] += (predatorStateBelief[node]/len(nodesToAddP))
                t = latestCurrentBelief-predatorStateBelief[node]
                predatorStateBelief[n] = (predatorStateBelief[n]/t)
        elif beliefForFaultyDrone == 1:
            if x == 0:
                t = 1-preyStateBelief[node]  + preyStateBelief[node]*0.1
                preyStateBelief[n] = (preyStateBelief[n]/t)
            else:
                t = latestCurrentBelief - predatorStateBelief[node] + predatorStateBelief[node]*0.1
                predatorStateBelief[n] = (predatorStateBelief[n]/t)
    if beliefForFaultyDrone == 1:
        if x == 0:
            t = 1-preyStateBelief[node]  + preyStateBelief[node]*0.1
            preyStateBelief[node] = preyStateBelief[node]*0.1 / t
            checkStateBelief(0)
        else:
            t = latestCurrentBelief - predatorStateBelief[node] + predatorStateBelief[node]*0.1
            predatorStateBelief[node] = predatorStateBelief[node]*0.1 / t
            checkStateBelief(1)

    if beliefForFaultyDrone == 0:
        if x == 0 :
            preyStateBelief[node] = float(0)
            checkStateBelief(0)
        elif beliefForFaultyDrone == 0:
            predatorStateBelief[node] = float(0)
            checkStateBelief(1)

def updateBeliefOnEntityMovement(x, distractedPredator = 1, nextStepLookup = 0):
    # print('In updateBeliefOnEntityMovement : ')
    global preyStateBelief, predatorStateBelief
    newBeliefState = {}
    for i in range(size):
        newBeliefState[i] = float(0)
    if x == 0:
        # nonZeroProbabilityIndices = findMaxProbability(preyStateBelief)
        nonZeroProbabilityIndices = getNonZeroProbabilityIndices(preyStateBelief)
        # print('nonZeroProbabilityIndices : ' + str(nonZeroProbabilityIndices))
        # Updating belief based on beleifs received
        # Agent 3's logic
        for node in nonZeroProbabilityIndices:
            adjNodes = g.getNextNodes(node)
            adjNodes.append(node)
            for j in adjNodes:
                # p = (1/len(adjNodes))/(1/preyStateBelief[node])
                p = (preyStateBelief[node]/len(adjNodes))
                if j not in newBeliefState.keys():
                    newBeliefState[j] = p
                else:
                    newBeliefState[j] += p
        if nextStepLookup == 1:
            return newBeliefState
        preyStateBelief = newBeliefState
    else:
        # checkStateBelief(1)
        # Agent 5, 6, 7, 8 Predator moving logic. Note: This logic is only for easily distracted predator movement
        nonZeroProbabilityIndices = getNonZeroProbabilityIndices(predatorStateBelief)
        # print('nonZeroProbabilityIndices : ' + str(nonZeroProbabilityIndices))
        # Updating belief based on previous beliefs received
        for node in nonZeroProbabilityIndices:
            adjNodes = g.getNextNodes(node)
            # adjNodes.append(node)
            if distractedPredator == 1:
                # 0.4 chance of predator taking a random next cell
                randomNodeProbab = 0.4 * predatorStateBelief[node]
                p = float(randomNodeProbab/len(adjNodes))
                for j in adjNodes:
                    if j not in newBeliefState.keys():
                        newBeliefState[j] = p
                    else:
                        newBeliefState[j] += p
                # 0.6 chance of predator taking the next cell as the next cell from BFS Path to agent
                # KV: nextCellToAgent = g.breadthFirstSearch(predator.getCurrNode(), agentPos)[0]
                nextCellToAgent = g.breadthFirstSearch(node, agentPos)[0]
                # print(nextCellToAgent)
                if len(nextCellToAgent) > 1:
                    newBeliefState[nextCellToAgent[1]] += ((1.0-0.4) * predatorStateBelief[node])
                else:
                    # quit()
                    newBeliefState[node] += ((1-0.4) * predatorStateBelief[node])
            else:
                # KV: nextCellToAgent = g.breadthFirstSearch(predator.getCurrNode(), agentPos)[0]
                nextCellToAgent = g.breadthFirstSearch(node, agentPos)[0]
                # print(nextCellToAgent)
                if len(nextCellToAgent) > 1:
                    newBeliefState[nextCellToAgent[1]] += predatorStateBelief[node]
                else:
                    # quit()
                    newBeliefState[node] += predatorStateBelief[node]
        if nextStepLookup == 1:
            return newBeliefState
        predatorStateBelief = newBeliefState
        checkStateBelief(1)

# Value of x : 0 = Prey Update, 1 = Predator Update, 3 = Prey and Predator both update
# surveyedNode : The node being surveyed. -1 value represents if node was 
def updateBelief(x, surveyedNode = -1, entityPresent = -1, beliefForFaultyDrone = 0):
    global preyStateBelief, predatorStateBelief, agentPos
    # Inititalizing newBeliefState as dummy dictionary to store new belief calculated
    newBeliefState = {}
    for i in range(size):
        newBeliefState[i] = float(0)
    if x == 0 or x == 3:
        checkStateBelief(0)
        if entityPresent == 0:
            checkStateBelief(0)
            distributeProbabilityFromCell(0, surveyedNode, beliefForFaultyDrone)
        elif entityPresent == 1:
            newBeliefState[surveyedNode] = float(1)
            preyStateBelief = newBeliefState
        elif entityPresent == -1:
            updateBeliefOnEntityMovement(0)
        checkStateBelief(0)
    elif x == 1 or x == 3:
        # preyStateBelief[agentPos] = 0
        if entityPresent == 0:
            distributeProbabilityFromCell(1, surveyedNode, beliefForFaultyDrone)
        elif entityPresent == 1:
            print('Predator found. Updating probability for Surveyed Node :',surveyedNode)
            # checkStateBelief(1)
            newBeliefState[surveyedNode] = float(1)
            predatorStateBelief = newBeliefState
            checkStateBelief(1)
        elif entityPresent == -1:
            updateBeliefOnEntityMovement(1)


## Agent 7 
def agentSevenMovement(defectiveDrone = 0, beliefForFaultyDrone = 0):
    global g, predator, prey, agentPos, predatorStateBelief, preyFoundCounter, predatorFoundCounter
    predatorPosKnown = isPredatorPosKnown()
    if predatorPosKnown:
        surveyedNode, preyPresent = surveyNode(0, defectiveDrone)
        if preyPresent == 1:
            preyFoundCounter += 1
        updateBelief(0, surveyedNode, preyPresent, beliefForFaultyDrone)
    else:
        surveyedNode, predatorPresent = surveyNode(1, defectiveDrone)
        if predatorPresent == 1:
            predatorFoundCounter += 1
        updateBelief(1, surveyedNode, predatorPresent, beliefForFaultyDrone)          # preyPresent = 1 if Prey present else 0
        if surveyedNode == prey.getCurrNode():
            updateBelief(0, surveyedNode, 1)
        else:
            updateBelief(0, surveyedNode, 0)
    nextNodesForAgent = g.getNextNodes(agentPos)
    print('nextNodesForAgent : '+str(nextNodesForAgent))
    # To get one of the node with highest probability of predator position (which is closest to the agentPosition) from predatorStateBelief
    predatorNode = random.choice(getDistanceWithClosestPredatorProximity())
    # To get one of the node with highest probability of prey position from preyStateBelief
    preyNode = random.choice(findMaxProbability(preyStateBelief))
    nearestPredatorNode, nearestPreyNode, farthestPredatorNode, farthestPreyNode, nextNodeAttributes = getNearestFarthestPreyPredator(nextNodesForAgent, preyNode=preyNode, predatorNode= predatorNode)
    
    nextAgentPos = agentOneDefinedSteps(nextNodeAttributes, nearestPreyNode, nearestPredatorNode)
    
    # If Agent's next position is changing, need to update probability accordingly so that probability summation remains 1
    if nextAgentPos == agentPos:
        agentPos = nextAgentPos
    else:
        if nextAgentPos != predator.getCurrNode():
            updateBelief(1, nextAgentPos, 0)
        if nextAgentPos != prey.getCurrNode():
            updateBelief(0, nextAgentPos, 0)
        agentPos = nextAgentPos
    showEntityPositions()

# Defines Agents 7 game in The Combined Partial Information Setting environment
def agentSeven(defectiveDrone = 0, beliefForFaultyDrone = 0):
    global g, predator, prey, agentPos, predatorStateBelief, preyFoundCounter, predatorFoundCounter
    tieBreaker = 0
    result = -1
    initializePredatorStateBelief()    # Setting the Predator belief
    initializePreyStateBelief()    # Setting the Prey belief
    # Movement of Agent 7, Prey and Predator until Agent 7 wins, or loses, or the system hangs
    while tieBreaker<5000:
        print('tieBreaker : '+ str(tieBreaker))
        agentSevenMovement(defectiveDrone, beliefForFaultyDrone)
        if (agentPos != predator.getCurrNode() and agentPos == prey.getCurrNode()):
            result = 0
            break
        preyPredatorMovement(easilyDistractedPredator=1)
        if (agentPos == predator.getCurrNode() and agentPos == prey.getCurrNode()):
            print('---- Agent in same position as Prey and Predator ----')
        elif (agentPos == prey.getCurrNode()):
            result = 0
            break
        elif (agentPos == predator.getCurrNode()):
            result = 1
            break
        updateBelief(0)
        updateBelief(1)
        tieBreaker += 1
    return [result, tieBreaker, preyFoundCounter, predatorFoundCounter]

def agentSevenLoop(defectiveDrone = 0, beliefForFaultyDrone = 0):
    global g, Predator, Prey, agentPos, preyFoundCounter, predatorFoundCounter
    result = []
    for i in range(10000):
        preyFoundCounter = 0
        predatorFoundCounter = 0
        g = create_env(g)
        agentPos = placeEntities(g)
        result.append(agentSeven(defectiveDrone, beliefForFaultyDrone))
        showEntityPositions()
    print('Agent 7 Result = ' + str(result))
    countTotal(result)
    writeToCSV(result, 7)





def customizedAgentMovement(preyPositionKnown, predatorPositionKnown, easilyDistractedPredator, defectiveDrone, beliefForFaultyDrone = 0, agentFaultyDroneUpdate = 0):
    global g, predator, prey, agentPos, preyStateBelief, predatorStateBelief, preyFoundCounter, predatorFoundCounter, surveyedNode
    if preyPositionKnown == 0 and predatorPositionKnown == 0:
        predatorPosKnown = isPredatorPosKnown()
        if predatorPosKnown:
            surveyedNode, preyPresent = surveyNode(0, defectiveDrone)
            if preyPresent == 1:
                preyFoundCounter += 1
            updateBelief(0, surveyedNode, preyPresent, beliefForFaultyDrone)
            surveyedNode = -1       # For use in Agent 9
        else:
            surveyedNode, predatorPresent = surveyNode(1, defectiveDrone)
            if predatorPresent == 1:
                predatorFoundCounter += 1
            updateBelief(1, surveyedNode, predatorPresent, beliefForFaultyDrone)          # preyPresent = 1 if Prey present else 0
            if surveyedNode == prey.getCurrNode():
                updateBelief(0, surveyedNode, 1)
            else:
                updateBelief(0, surveyedNode, 0)
            if predatorPresent == 0:
                surveyedNode = -1       # For use in Agent 9
    elif preyPositionKnown == 0 and predatorPositionKnown == 1:
        surveyedNode, preyPresent = surveyNode(0, defectiveDrone)
        if preyPresent == 1:
            preyFoundCounter += 1
        updateBelief(0, surveyedNode, preyPresent, beliefForFaultyDrone)
        surveyedNode = -1       # For use in Agent 9
    elif preyPositionKnown == 1 and predatorPositionKnown == 0:
        surveyedNode, predatorPresent = surveyNode(1, defectiveDrone)
        if predatorPresent == 1:
            predatorFoundCounter += 1
        updateBelief(1, surveyedNode, predatorPresent, beliefForFaultyDrone)
        if predatorPresent == 0:
            surveyedNode = -1       # For use in Agent 9
    preyMidStateBelief, predatorMidStateBelief = updateBeliefNextStep(easilyDistractedPredator, nextStepLookup= 1)
    # nextNodesForAgent = g.getNextNodes(agentPos)
    # print('nextNodesForAgent : '+str(nextNodesForAgent))
    # # To get one of the node with highest probability of predator position (which is closest to the agentPosition) from predatorStateBelief
    # predatorNode = random.choice(getDistanceWithClosestPredatorProximity())
    # # To get one of the node with highest probability of prey position from preyStateBelief
    # preyNode = random.choice(findMaxProbability(preyStateBelief))
    # nearestPredatorNode, nearestPreyNode, farthestPredatorNode, farthestPreyNode, nextNodeAttributes = getNearestFarthestPreyPredator(nextNodesForAgent, preyNode=preyNode, predatorNode= predatorNode)
    
    nextAgentPos = agentCustomizedDefinedSteps(predatorPositionKnown = predatorPositionKnown, preyMidStateBelief= preyMidStateBelief, predatorMidStateBelief= predatorMidStateBelief, agentFaultyDroneUpdate = 0)
    # nextAgentPos = agentOneDefinedSteps(nextNodeAttributes, nearestPreyNode, nearestPredatorNode)
    
    # If Agent's next position is changing, need to update probability accordingly so that probability summation remains 1
    if nextAgentPos != predator.getCurrNode():
        updateBelief(1, nextAgentPos, 0)
    if nextAgentPos != prey.getCurrNode():
        updateBelief(0, nextAgentPos, 0)

    agentPos = nextAgentPos
    
    # agentPos = g.breadthFirstSearch(agentPos, bestCellToGetTo)[0][1]
    # pathTowardsPrey = g.breadthFirstSearch(agentPos, bestCellToGetTo)[0][1]
    # if predator.getCurrNode() in pathTowardsPrey:
    #     pass

    # print('----Agent Moved: ')
    # showEntityPositions()


def customizedAgent(preyPositionKnown, predatorPositionKnown, easilyDistractedPredator, defectiveDrone, beliefForFaultyDrone=0, surveyOrMoveAgent = 0, agentFaultyDroneUpdate = 0):
    global g, predator, prey, agentPos, preyFoundCounter, predatorFoundCounter
    tieBreaker = 0
    result = -1
    if predatorPositionKnown == 0:
        initializePredatorStateBelief()    # Setting the Predator belief
    if preyPositionKnown == 0:
        initializePreyStateBelief(preyPositionKnown)    # Setting the Prey belief
    while tieBreaker<5000:
        if preyPositionKnown == 1:
            initializePreyStateBelief(preyPositionKnown)
        if predatorPositionKnown == 1:
            initializePredatorStateBelief(predatorPositionKnown)
        if surveyOrMoveAgent == 1:
            surveyOrMoveAgentMovement(preyPositionKnown, predatorPositionKnown, easilyDistractedPredator, defectiveDrone, beliefForFaultyDrone)
        else:
            customizedAgentMovement(preyPositionKnown, predatorPositionKnown, easilyDistractedPredator, defectiveDrone, beliefForFaultyDrone, agentFaultyDroneUpdate)
        if (agentPos != predator.getCurrNode() and agentPos == prey.getCurrNode()):
            result = 0
            break
        preyPredatorMovement(easilyDistractedPredator)
        updateBeliefNextStep(easilyDistractedPredator)
        showEntityPositions()
        if (agentPos == predator.getCurrNode() and agentPos == prey.getCurrNode()):
            print('---- Agent in same position as Prey and Predator ----')
        elif (agentPos == prey.getCurrNode()):
            result = 0
            break
        elif (agentPos == predator.getCurrNode()):
            result = 1
            break
        tieBreaker += 1
    return [result, tieBreaker, preyFoundCounter, predatorFoundCounter]


def customizedAgentLoop(preyPositionKnown, predatorPositionKnown, easilyDistractedPredator, defectiveDrone, agentNo, beliefForFaultyDrone = 0, surveyOrMoveAgent = 0, agentFaultyDroneUpdate = 0):
    global g, Predator, Prey, agentPos, preyFoundCounter, predatorFoundCounter
    result = []
    for i in range(10000):
        preyFoundCounter = 0
        predatorFoundCounter = 0
        print('Agent run: ',i+1)
        g = create_env(g)
        agentPos = placeEntities(g)
        showEntityPositions()
        result.append(customizedAgent(preyPositionKnown, predatorPositionKnown, easilyDistractedPredator, defectiveDrone, beliefForFaultyDrone, surveyOrMoveAgent, agentFaultyDroneUpdate))
        showEntityPositions()
    # print('Customized Agent Result = ' + str(result))
    # countTotal(result)
    writeToCSV(result, agentNo)
    return result


def countTotal(result):
    numberOfWins = 0
    numberOfLoses = 0
    numberOfTies = 0
    for i in result[0]:
        if i == 0:
            numberOfWins += 1
        elif i == 1:
            numberOfLoses += 1
        elif i == -1:
            numberOfTies += 1
    print('Total - 0 = '+str(numberOfWins)+', 1 = '+str(numberOfLoses)+', -1 = '+str(numberOfTies))

def generateGraph():
    global genG, g
    genG = nx.Graph()
    for i in range(size):
        genG.add_node(i)
    for i in range(size):
        nodes = g.getNextNodes(i)
        for n in nodes:
            genG.add_edge(i, n)


