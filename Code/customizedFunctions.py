import random
import Code.Settings as gVar
import csv


# Setting the Prey belief:
def initializePreyStateBelief(preyKnown = 0):
    global preyStateBelief, prey, agentPos, size
    b = 1/(size-1)
    for i in range(size):
        if i != agentPos and preyKnown==0:
            preyStateBelief[i] = b
        elif i == prey.getCurrNode() and preyKnown==1:
            preyStateBelief[i] = float(1)
        else:
            preyStateBelief[i] = float(0)

# Initializing the Predator belief
def initializePredatorStateBelief(nextStepUpdate = 0):
    global predatorStateBelief, predator
    for i in range(size):
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
def create_env():
    gVar.g.randomLinkandEdges()

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
    # print('Agent Position : ' + str(agentPos))
    # print('Predator Position : ' + str(predator.getCurrNode()))
    # print('Prey Position : ' + str(prey.getCurrNode()))
    print('Agent Position : ' + str(gVar.agentPos)+'; Predator Position : ' + str(gVar.predator.getCurrNode())+'; Prey Position : ' + str(gVar.prey.getCurrNode()))

# To initiate movement of the Prey and Predator in the current timestamp
def preyPredatorMovement(easilyDistractedPredator = 1):
    # Movement of Prey
    gVar.prey.move(gVar.g.getNextNodes(gVar.prey.currNode))
    if easilyDistractedPredator == 0:
        gVar.predator.move(gVar.g.breadthFirstSearch(gVar.predator.getCurrNode(), gVar.agentPos)[0])    # Taking only the first argument (path) as received from g.breadthFirstSearch()
    else:
        gVar.predator.move(gVar.g.breadthFirstSearch(gVar.predator.getCurrNode(), gVar.agentPos)[0], 1)


def isPredatorPosKnown():
    global predatorStateBelief
    p = findMaxProbability(predatorStateBelief)
    if predatorStateBelief[p[0]] == 1:
        return True
    else:
        return False

# Returns nodes with maximum probability/number in the beliefState/dictionary received
def findMaxProbability(beliefState, nextStep = 0):
    # maxB = [list of indices with maximum probability]
    # Initializing maxB with least values
    maxB = -1
    maxProbabilityIndices = []
    for i in beliefState:
        if i == agentPos and nextStep == 0:
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
        for i in preyStateBelief:
            m+= preyStateBelief[i]
        # print('preyStateBelief : '+str(preyStateBelief))
        # print('preyStateBelief Probability Sum : '+str(m))
        if m == 0 or m <0.99 or m>1.01:
            quit()
    else:
        for i in predatorStateBelief:
            m+= predatorStateBelief[i]
        # print('predatorStateBelief : '+str(predatorStateBelief))
        # print('predatorStateBelief Probability Sum : '+str(m))
        if m == 0 or m <0.9999 or m>1.0001:
            print('Fail')
            #quit()

def getStateBelief(x):
    m = float(0)
    if x == 0:
        for i in preyStateBelief:
            m+= preyStateBelief[i]
    else:
        for i in predatorStateBelief:
            m+= predatorStateBelief[i]
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