import random

class Prey:
    def __init__(self) -> None:
        print('called')
        pass

    def setNode(self, currNd):
        self.currNode = currNd

    def getCurrNode(self):
        return self.currNode
    
    # Prey moves to one of the next nodes randomly: i.e. (nextNodes received + currNode)
    def move(self, nextNodes):
        nextNodes.append(self.currNode)
        self.currNode = random.choice(nextNodes)