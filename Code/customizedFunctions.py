import random
import Code.Settings as gVar
import csv
import pickle


# Setting the Prey belief:
def initializePreyStateBelief(preyKnown = 0):
    b = 1/(gVar.size-1)
    for i in range(gVar.size):
        if i != gVar.agentPos and preyKnown==0:
            gVar.preyStateBelief[i] = b
        elif i == gVar.prey.getCurrNode() and preyKnown==1:
            gVar.preyStateBelief[i] = float(1)
        else:
            gVar.preyStateBelief[i] = float(0)

# Initializing the Predator belief
def initializePredatorStateBelief(nextStepUpdate = 0):
    global predatorStateBelief, predator
    for i in range(gVar.size):
        if i!=predator.currNode:
            predatorStateBelief[i] = float(0)
        else:
            predatorStateBelief[i] = float(1)

# Get Random object among the objects passed
def getRandomObjectAmongPassed(nodes):
    nodeNrSelected = random.choice(range(len(nodes)))
    return nodes[nodeNrSelected]

# To get first Tuple elements among the objects passed
def fetchFirstTupleElement(nodes):
    res = []
    for i in nodes:
        res.append(nodes[0])
    return res

# Function to create the Tree
def create_env(newEnv = 0):
    if newEnv == 0:
        gVar.g.randomLinkandEdges()
    else:
        with open('/Users/mitulshah/Desktop/AI Project3/graph4.pickle', 'rb') as f:
            dataG = pickle.load(f)
        gVar.g.importFromData(dataG)

def populateUtilityAtTimeT(vModel=0):
    # with open('/Users/kunjanvaghela/Projects/Project 3 Better Smarter Faster/BetterSmarterFaster/ImportData/UtilitiesFinal.csv', newline='') as f:
    with open('/Users/mitulshah/Desktop/AI Project3/VModelOutput2_NoIndex.csv', newline='') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            # print(row)
            if i > 0:
                # csv_list = [[val.strip() for val in r.split(",")] for r in f.readlines()]
                gVar.utilityAtTimeT[(int(row[1]),int(row[2]),int(row[3]))] = float(row[6])
                # csv_list = [val for val in row[0].split(",")]
                # for r in len(csv_list):
                #     # if row0[r]
                #     csv_list[r] = int(csv_list[r][1])
                pass
            i += 1
    f.close()

    if vModel == 1:
        print(vModel)
        for i in range(gVar.size):
            for j in range(gVar.size):
                for k in range(gVar.size):
                    # gVar.states.append((i,j,k))
                    # gVar.utilityOfNextAction[(i,j,k)] = {}
                    if i == k:
                        gVar.utilityAtTimeT[(i,j,k)] = -float('inf')
                    elif i == j:
                        gVar.utilityAtTimeT[(i,j,k)] = 0.
                    elif k in gVar.g.getNextNodes(i):
                        gVar.utilityAtTimeT[(i,j,k)] = -float('inf')

        #Read other Utility values from oldFile.
        with open('/Users/mitulshah/Desktop/AI Project3/UtilitiesFinal.csv', newline='') as f2:
            reader2 = csv.reader(f2)
            i = 0
            for row in reader2:
                # print(row)
                if i > 0:
                    if float(row[6]) == float('-inf') or row[6] == 0.:
                        gVar.utilityAtTimeT[(int(row[1]),int(row[2]),int(row[3]))] = float(row[6])
                i += 1
        f2.close()

    gVar.utilityAtTimeTMinus1 = gVar.utilityAtTimeT.copy()

# Spawns Prey, Predator and Agent. Prey and Predator are spawned randomly, while Agent is spawned in any place other than Prey and Predator
def placeEntities(g):
    gVar.prey.setNode(g.getRandomNode())
    gVar.predator.setNode(g.getRandomNode())
    gVar.agentPos = gVar.g.getRandomNode()
    while (gVar.agentPos == gVar.predator.currNode or gVar.agentPos== gVar.prey.currNode):
        gVar.agentPos = gVar.g.getRandomNode()
    return gVar.agentPos

# To show position of all the entities:
def showEntityPositions():
    print('Agent Position : ' + str(gVar.agentPos)+'; Predator Position : ' + str(gVar.predator.getCurrNode())+'; Prey Position : ' + str(gVar.prey.getCurrNode()))

# To initiate movement of the Prey and Predator in the current timestamp
#Do not move them together.
def preyPredatorMovement(easilyDistractedPredator = 1):
    # Movement of Prey
    gVar.prey.move(gVar.g.getNextNodes(gVar.prey.currNode))
    if easilyDistractedPredator == 0:
        gVar.predator.move(gVar.g.breadthFirstSearch(gVar.predator.getCurrNode(), gVar.agentPos)[0])    # Taking only the first argument (path) as received from g.breadthFirstSearch()
    else:
        gVar.predator.move(gVar.g.breadthFirstSearch(gVar.predator.getCurrNode(), gVar.agentPos)[0], 1)


def preyMovement():
    # Movement of Prey
    gVar.prey.move(gVar.g.getNextNodes(gVar.prey.currNode))

def predatorMovement(easilyDistractedPredator = 1):
    if easilyDistractedPredator == 0:
        gVar.predator.move(gVar.g.breadthFirstSearch(gVar.predator.getCurrNode(), gVar.agentPos)[0])    # Taking only the first argument (path) as received from g.breadthFirstSearch()
    else:
        gVar.predator.move(gVar.g.breadthFirstSearch(gVar.predator.getCurrNode(), gVar.agentPos)[0], 1)

def isPredatorPosKnown():
    p = findMaxProbability(gVar.predatorStateBelief)
    if gVar.predatorStateBelief[p[0]] == 1:
        return True
    else:
        return False

# Returns nodes with maximum probability/number in the beliefState/dictionary received
def findMaxProbability(beliefState, nextStep = 0):
    # Initializing maxB with least values
    maxB = -1
    maxProbabilityIndices = []
    for i in beliefState:
        if i == gVar.agentPos and nextStep == 0:
            continue
        if beliefState[i] > maxB:
            maxB = beliefState[i]
            maxProbabilityIndices = [i]
        elif beliefState[i] == maxB:
            maxProbabilityIndices.append(i)
    return maxProbabilityIndices

# Returns cells with non-zero probability in the given belief states
def getNonZeroProbabilityIndices(beliefState):
    res = []
    for i in beliefState:
        if beliefState[i] > 0:
            res.append(i)
    return res

# x: Defines which stateBelief to check 0 = preyStateBelief, 1 = predatorStateBelief
def checkStateBelief(x):
    m = float(0)
    if x == 0:
        for i in gVar.preyStateBelief:
            m+= gVar.preyStateBelief[i]
        if m == 0 or m <0.99 or m>1.01:
            quit()
    else:
        for i in gVar.predatorStateBelief:
            m+= gVar.predatorStateBelief[i]
        if m == 0 or m <0.9999 or m>1.0001:
            print('Fail')
            #quit()

def getStateBelief(x):
    m = float(0)
    if x == 0:
        for i in gVar.preyStateBelief:
            m+= gVar.preyStateBelief[i]
    else:
        for i in gVar.predatorStateBelief:
            m+= gVar.predatorStateBelief[i]
    return m


# input: List containing each element as list of Win/Loss/Tie, No. of steps, PreyFoundCounter, PredatorFoundCounter
# def writeToCSV(data):
#     with open('Utilities.csv', 'w', newline='') as file:
#         writer = csv.writer(file, delimiter='')
#         # if agentNo <= 2:
#         #     writer.writerow(['Status','Steps'])
#         # elif agentNo <= 3:
#         #     writer.writerow(['Status','Steps','PreyFoundCounter'])
#         # elif agentNo >= 4:
#         #     writer.writerow(['Status','Steps','PreyFoundCounter','PredatorFoundCounter'])
#         # elif agentNo > 6:
#         #     writer.writerow(['Status','Steps','PreyFoundCounter','PredatorFoundCounter'])
#         for i in data:
#             writer.writerow(str(i)+' : '+(str(data[i])))
#         file.close()
#     # print("Successfully written to file {}", 'a_' + str(agentNo) + '.csv')
#     print("Successfully written to file.")

def writeToCSV(data):
    with open('Utilities.txt', 'w') as file:
        for i in data:
            file.write(str(i)+' : '+(str(data[i]))+'\n')
        file.close()
    # print("Successfully written to file {}", 'a_' + str(agentNo) + '.csv')
    print("Successfully written to file.")

def writeToFile(data, filename):
    filename = str(filename + '.txt')
    with open(filename, 'w') as file:
        for i in data:
            file.write(str(i)+' : '+(str(data[i]))+'\n')
        file.close()
    # print("Successfully written to file {}", 'a_' + str(agentNo) + '.csv')
    print("Successfully written to file.")

def writeListToFile(data, filename):
    filename = str(filename + '.txt')
    with open(filename, 'w') as file:
        for i in data:
            file.write(str(i)+'\n')
        file.close()
    # print("Successfully written to file {}", 'a_' + str(agentNo) + '.csv')
    print("Successfully written to file.")






##### Survey Logic
# This function will return the max predator probability belief node which is nearest to the agent among other max predator probability belief nodes
def getDistanceWithClosestPredatorProximity(predatorStateBelief = gVar.predatorStateBelief):
    pos = findMaxProbability(predatorStateBelief)
    minDist = -1
    minDistList = []
    for p in pos:
        if p == gVar.agentPos:
            continue
        pathReceived, distance = gVar.g.breadthFirstSearch(gVar.agentPos, p)
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
    choiceDefective = random.random()
    if surveyFor == 0:
        pos = findMaxProbability(gVar.preyStateBelief)
        positionToCheck = random.choice(pos)
        if defectiveDrone==1 and choiceDefective < 0.1:
            return (positionToCheck, 0)
        if positionToCheck == gVar.prey.getCurrNode():
            return (positionToCheck, 1)
        else:
            return (positionToCheck, 0)
    else:
        minDistList = getDistanceWithClosestPredatorProximity()
        positionToCheck = random.choice(minDistList)
        if defectiveDrone==1 and choiceDefective < 0.1:
            return (positionToCheck, 0)
        if positionToCheck == gVar.predator.getCurrNode():
            return (positionToCheck, 1)
        else:
            return (positionToCheck, 0)

# Distributes probability from the input cell into other cells with non-zero probability values
# x : 0 = Prey Update, 1 = Predator Update
def distributeProbabilityFromCell(x, node, beliefForFaultyDrone = 0):
    if x == 0:
        checkStateBelief(0)
        nodesToAddP = [i for i in gVar.preyStateBelief if gVar.preyStateBelief[i] > 0]
    else:
        checkStateBelief(1)
        latestCurrentBelief = getStateBelief(1)
        nodesToAddP = [i for i in gVar.predatorStateBelief if gVar.predatorStateBelief[i] > 0]
    # print('nodesToAddP : '+str(nodesToAddP))
    if node in nodesToAddP:
        nodesToAddP.remove(node)
    for n in nodesToAddP:
        if beliefForFaultyDrone == 0:
            if x == 0:
                # preyStateBelief[n] += (preyStateBelief[node]/len(nodesToAddP))
                t = 1-gVar.preyStateBelief[node]
                # preyStateBelief[n] += (preyStateBelief[n]*t)
                gVar.preyStateBelief[n] = (gVar.preyStateBelief[n]/t)
                # checkStateBelief(0)
            else:
                # predatorStateBelief[n] += (predatorStateBelief[node]/len(nodesToAddP))
                t = latestCurrentBelief-gVar.predatorStateBelief[node]
                gVar.predatorStateBelief[n] = (gVar.predatorStateBelief[n]/t)
        elif beliefForFaultyDrone == 1:
            if x == 0:
                t = 1-gVar.preyStateBelief[node]  + gVar.preyStateBelief[node]*0.1
                gVar.preyStateBelief[n] = (gVar.preyStateBelief[n]/t)
            else:
                t = latestCurrentBelief - gVar.predatorStateBelief[node] + gVar.predatorStateBelief[node]*0.1
                gVar.predatorStateBelief[n] = (gVar.predatorStateBelief[n]/t)
    if beliefForFaultyDrone == 1:
        if x == 0:
            t = 1-gVar.preyStateBelief[node]  + gVar.preyStateBelief[node]*0.1
            gVar.preyStateBelief[node] = gVar.preyStateBelief[node]*0.1 / t
            checkStateBelief(0)
        else:
            t = latestCurrentBelief - gVar.predatorStateBelief[node] + gVar.predatorStateBelief[node]*0.1
            gVar.predatorStateBelief[node] = gVar.predatorStateBelief[node]*0.1 / t
            checkStateBelief(1)

    if beliefForFaultyDrone == 0:
        if x == 0 :
            gVar.preyStateBelief[node] = float(0)
            checkStateBelief(0)
        elif beliefForFaultyDrone == 0:
            gVar.predatorStateBelief[node] = float(0)
            checkStateBelief(1)

def updateBeliefOnEntityMovement(x, distractedPredator = 1, nextStepLookup = 0):
    newBeliefState = {}
    for i in range(gVar.size):
        newBeliefState[i] = float(0)
    if x == 0:
        # nonZeroProbabilityIndices = findMaxProbability(preyStateBelief)
        nonZeroProbabilityIndices = getNonZeroProbabilityIndices(gVar.preyStateBelief)
        # print('nonZeroProbabilityIndices : ' + str(nonZeroProbabilityIndices))
        # Updating belief based on beleifs received
        # Agent 3's logic
        for node in nonZeroProbabilityIndices:
            adjNodes = gVar.g.getNextNodes(node)
            adjNodes.append(node)
            for j in adjNodes:
                # p = (1/len(adjNodes))/(1/preyStateBelief[node])
                p = (gVar.preyStateBelief[node]/len(adjNodes))
                if j not in newBeliefState.keys():
                    newBeliefState[j] = p
                else:
                    newBeliefState[j] += p
        if nextStepLookup == 1:
            return newBeliefState
        gVar.preyStateBelief = newBeliefState
    else:
        # Agent 5, 6, 7, 8 Predator moving logic. Note: This logic is only for easily distracted predator movement
        nonZeroProbabilityIndices = getNonZeroProbabilityIndices(gVar.predatorStateBelief)
        # Updating belief based on previous beliefs received
        for node in nonZeroProbabilityIndices:
            adjNodes = gVar.g.getNextNodes(node)
            # adjNodes.append(node)
            if distractedPredator == 1:
                # 0.4 chance of predator taking a random next cell
                randomNodeProbab = 0.4 * gVar.predatorStateBelief[node]
                p = float(randomNodeProbab/len(adjNodes))
                for j in adjNodes:
                    if j not in newBeliefState.keys():
                        newBeliefState[j] = p
                    else:
                        newBeliefState[j] += p
                # 0.6 chance of predator taking the next cell as the next cell from BFS Path to agent
                # KV: nextCellToAgent = g.breadthFirstSearch(predator.getCurrNode(), agentPos)[0]
                nextCellToAgent = gVar.g.breadthFirstSearch(node, gVar.agentPos)[0]
                # print(nextCellToAgent)
                if len(nextCellToAgent) > 1:
                    newBeliefState[nextCellToAgent[1]] += ((1.0-0.4) * gVar.predatorStateBelief[node])
                else:
                    # quit()
                    newBeliefState[node] += ((1-0.4) * gVar.predatorStateBelief[node])
            else:
                # KV: nextCellToAgent = g.breadthFirstSearch(predator.getCurrNode(), agentPos)[0]
                nextCellToAgent = gVar.g.breadthFirstSearch(node, gVar.agentPos)[0]
                # print(nextCellToAgent)
                if len(nextCellToAgent) > 1:
                    newBeliefState[nextCellToAgent[1]] += gVar.predatorStateBelief[node]
                else:
                    # quit()
                    newBeliefState[node] += gVar.predatorStateBelief[node]
        if nextStepLookup == 1:
            return newBeliefState
        gVar.predatorStateBelief = newBeliefState
        checkStateBelief(1)

# Value of x : 0 = Prey Update, 1 = Predator Update, 3 = Prey and Predator both update
# surveyedNode : The node being surveyed. -1 value represents if node was 
def updateBelief(x, surveyedNode = -1, entityPresent = -1, beliefForFaultyDrone = 0):
    # Inititalizing newBeliefState as dummy dictionary to store new belief calculated
    newBeliefState = {}
    for i in range(gVar.size):
        newBeliefState[i] = float(0)
    if x == 0 or x == 3:
        checkStateBelief(0)
        if entityPresent == 0:
            checkStateBelief(0)
            distributeProbabilityFromCell(0, surveyedNode, beliefForFaultyDrone)
        elif entityPresent == 1:
            newBeliefState[surveyedNode] = float(1)
            gVar.preyStateBelief = newBeliefState
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
            gVar.predatorStateBelief = newBeliefState
            checkStateBelief(1)
        elif entityPresent == -1:
            updateBeliefOnEntityMovement(1)
