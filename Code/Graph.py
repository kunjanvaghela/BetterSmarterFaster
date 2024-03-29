import matplotlib.pyplot as plt
import random
import numpy as np
import Code.Settings as gVar


class Graph(object):
    # Initialize the matrix
    def __init__(self, size):
        self.adjMatrix = []
        for i in range(size):
            self.adjMatrix.append([0 for i in range(size)])
        self.size = size
        # print('matrix:',self.adjMatrix)

    # Add edges
    def add_edge(self, node1, node2):
        if node1 == node2:
            print("Same vertex %d and %d" % (node1, node2))
        self.adjMatrix[node1][node2] = 1
        self.adjMatrix[node2][node1] = 1

    def checkdegree(self,node1):
        degree=0
        #for i in range(size): 
        for j in range(self.size):
            #print('node1:',node1)
            #print('j:',j)
            if((self.adjMatrix[node1][j])==1):
                degree+=1
        #print('degree:',degree,'::node1',node1)
        return degree

    def add_randomEdge(self, node1):
            inclusionlist= list(range(node1-5,node1+5))
            inclusionlist.remove(node1)
            inclusionlist.remove(node1+1)
            inclusionlist.remove(node1-1)
            # print(inclusionlist)
            #randomnode= choice([i for i in range(0, size) if i not in exceptionlist])
            while(self.checkdegree(node1)<3 and len(inclusionlist)>0):
                randomnodeseed= random.choice(inclusionlist)
                # print('::node1',node1,':randomnode:',randomnodeseed)
                randomnode=randomnodeseed
                if randomnodeseed>(self.size-6):
                    randomnode=randomnodeseed-(self.size)
                if randomnodeseed<0:
                    randomnode=self.size+randomnodeseed
                #print('::node1',node1,'g.checkdegree(node1)',g.checkdegree(node1),'randomnode:',randomnode,'::g.checkdegree(randomnode)::',g.checkdegree(randomnode))
                
                if self.checkdegree(randomnode)<3:
                    #print('::node1',node1,'g.checkdegree(node1)',g.checkdegree(node1),'randomnode:',randomnode,'::g.checkdegree(randomnode)::',g.checkdegree(randomnode))
                    if node1 == randomnode:
                        print("Same vertex %d and %d" % (node1, randomnode))
                    self.adjMatrix[node1][randomnode] = 1
                    self.adjMatrix[randomnode][node1] = 1

                inclusionlist.remove(randomnodeseed)
                
        # Remove edges
    def remove_edge(self, node1, node2):
        if self.adjMatrix[node1][node2] == 0:
            print("No edge between %d and %d" % (node1, node2))
            return
        self.adjMatrix[node1][node2] = 0
        self.adjMatrix[node2][node1] = 0

    def __len__(self):
        return self.size

    # Print the matrix
    def print_matrix(self):
        print('matrix:',self.adjMatrix)
        plt.matshow(np.asarray(self.adjMatrix))
        plt.show()

    def randomLinkandEdges(self):
        for i in range(0, self.size):
          if i==0:
            self.add_edge(0, self.size-1)
            self.add_edge(0,i+1)
          elif i== self.size-1:
            break
          else:
            self.add_edge(i,i+1)
         #g.print_matrix()
        for i in range (self.size):
            if(self.checkdegree(i)<3):
                self.add_randomEdge(i)
    
    def importFromData(self, data):
        self.adjMatrix = data
    
    # returns a random node, this is used to get random nodes to spawn Prey and Predator
    def getRandomNode(self):
        return random.choice(range(self.size))
    
    # To get the nodes adjacent/connected to the node passed in the adjacency matrix. 
    def getNextNodes(self, node):
        res = []
        for j in range(self.size):
            if((self.adjMatrix[node][j])==1):
                res.append(j)
        # print('Next nodes : '+str(res))
        return res
    
    # For retrieving Path found from BFS Algo
    def print_bfs_path(self, childToParentMapping, source, destination):
        # print(childToParentMapping)
        curr = destination
        path = []
        path.append(destination)
        while curr != source:
            curr = childToParentMapping[curr]
            path.append(curr)
        path.reverse()
        return path

    # Modify BFS later to get multiple shortest paths: Query - Right now
    def breadthFirstSearch(self, source, destination):      # Takes source and destination nodes, and returns tuple of (list BFSPath, int distance)
        fringe = [source]
        # visited = set()
        childToParentMap = {}
        distanceMapping = {}
        shortestPathFound = False
        for i in range(self.size):
            distanceMapping[i] = -1
        distanceMapping[source] = 0
        while len(fringe) != 0:
            # print('Fringe : '+ str(fringe))
            currCell = fringe.pop(0)
            if currCell == destination:
                # return currCell
                # print('currCell == destination : ' + str(distanceMapping))
                # print('childToParentMap : '+ str(childToParentMap))
                shortestPathFound = True
                break
            nextCells = self.getNextNodes(currCell)
            for i in nextCells:
                if distanceMapping[i] == -1 and shortestPathFound == False:
                    distanceMapping[i] = distanceMapping[currCell] + 1
                    fringe.append(i)
                    childToParentMap[i] = currCell
                # if i not in visited:
                #     fringe.append(i)
                #     childToParentMap[i] = currCell
            # visited.add(currCell)
        # print(distanceMapping)
        pathAndDistance = (self.print_bfs_path(childToParentMap, source, destination),distanceMapping[destination])
        return (pathAndDistance)
