import random
import Code.Settings as gVar

class Predator:
    def __init__(self) -> None:
        pass

    def setNode(self, currNd):
        self.currNode = currNd
        
    def getCurrNode(self):
        return self.currNode

    # Predator moves towards the next node of the BFS path received
    # if easilyDistractedPredator = 0: predator will move in BFS path towards Agent, otherwise will move either randomly (0.4) or towards BFS path towards Agent (0.6)
    def move(self, bfsPathReceived, easilyDistractedPredator = 1):
        # print('Predator to Agent BFS Path : '+ str(bfsPathReceived))
        global g
        choice = random.random()
        if (easilyDistractedPredator == 0) or (easilyDistractedPredator != 0 and choice < 0.6):
            if len(bfsPathReceived) > 1:
                self.currNode = bfsPathReceived[1]
            elif len(bfsPathReceived) == 1:
                self.currNode = bfsPathReceived[0]
        else:
            nextNodes = gVar.g.getNextNodes(self.currNode)
            # nextNodes.append(self.currNode)
            self.currNode = random.choice(nextNodes)