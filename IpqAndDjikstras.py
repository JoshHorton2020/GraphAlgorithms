'''
requires networkx, argparse
requires python 3.6+ (can get with anaconda or elsewhere, note standard python with mac is python 2)
pip install networkx
pip install argparse
'''

import argparse
import networkx as nx
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
args = parser.parse_args() # no arguments but im leaving this here

'''
Problem 1
Implement the indexed priority queue data structure backed by a list based min-heap 
and a dict based index.
'''
class IndexedPriorityQueue:
    def __init__(self):
        self.min_heap = []
        self.index = {}

    def push(self, key, value):
        # push to end of list, then check is parent val is greater and if so then swap
        self.min_heap.append(value)
        self.index[key] = len(self.min_heap) - 1
        parent = self.min_heap[(len(self.min_heap) - 1) // 2]
        if (value != float('inf') or parent > value): 
            self.__heapify_up(key)

    def popmin(self):
        #double check to make sure heap not empty
        if not self.min_heap:
            return
        min_node = lastElementKey = list(filter(lambda x: self.index[x] == 0, self.index))[0]

        #first remove the root node and replace it with the last element
        self.min_heap[0] = self.min_heap[-1]
        lastElementKey = list(filter(lambda x: self.index[x] == len(self.min_heap)-1, self.index))[0]

        self.index[lastElementKey] = 0
        self.min_heap.pop()

        self.__heapify_down(lastElementKey)
        #fix the heap property by heapifying down accordingly
        return min_node
    
    def peek(self):
        #return min element, which should always be at 0
        return self.min_heap[0]
    
    def decrease_key(self, key, new_value):
        i = self.index[key]
        self.min_heap[i] = new_value

        #find the index of the parents and the children so that you can check if heap has been violated
        if i != 0:
            parent = self.min_heap[(i-1) // 2]
        else: 
            parent = None

        child1, child2 = None, None 
        if 2*i + 1 < len(self.min_heap):
            child1 = self.min_heap[(2*i) + 1]
        if 2*i + 2 < len(self.min_heap):
            child2 = self.min_heap[(2*i) + 2]

        #check if we violate that heap 
        if parent != None and parent > self.min_heap[i]:
            self.__heapify_up(key)
        elif (child1 != None and child1 < self.min_heap[i]) or (child2 != None and child2 < self.min_heap[i]):
            self.__heapify_down(key)

    def __heapify_up(self, key):
        #we already check if we need to be here before entering, so we can proceed by finding and swapping the elements and indices
        currentNodeIndex = self.index[key]
        parentNodeIndex = (currentNodeIndex - 1) // 2
        parentNodeKey = list(filter(lambda x: self.index[x] == parentNodeIndex, self.index))[0]
    
        currentNodeValue = self.min_heap[currentNodeIndex]
        parentNodeValue = self.min_heap[parentNodeIndex]

        self.min_heap[currentNodeIndex] = parentNodeValue  
        self.min_heap[parentNodeIndex] = currentNodeValue

        self.index[key] = parentNodeIndex
        self.index[parentNodeKey] = currentNodeIndex

        #check if parent in heap still violates the heap property and recurse?
        if currentNodeIndex > 0 and self.min_heap[currentNodeIndex] < self.min_heap[parentNodeIndex]:
            self.__heapify_up(key)
        
    def __heapify_down(self, key):
        #basically works same as heap up where we check if either child can be swapped with, 
        #then just arbitrarily swap with the smaller one and see if we still violate the heap prop, 
        #finally if we do violate it then we recurse back and do the same thing
        print(self.min_heap)
        currentNodeIndex = self.index[key]
        leftChildIndex = (2 * currentNodeIndex) + 1
        rightChildIndex = (2 * currentNodeIndex) + 2
        smallestIndex = currentNodeIndex

        #find smallest child to swap with 
        if leftChildIndex < len(self.min_heap) and self.min_heap[leftChildIndex] < self.min_heap[smallestIndex]:
            smallestIndex = leftChildIndex
        if rightChildIndex < len(self.min_heap) and self.min_heap[rightChildIndex] < self.min_heap[smallestIndex]:
            smallestIndex = rightChildIndex

        if smallestIndex != currentNodeIndex:
            smallestKey = list(filter(lambda x: self.index[x] == smallestIndex, self.index))[0]

            temp = self.min_heap[currentNodeIndex]
            self.min_heap[currentNodeIndex] = self.min_heap[smallestIndex]
            self.min_heap[smallestIndex] = temp
            
            self.index[key] = smallestIndex
            self.index[smallestKey] = currentNodeIndex

            #keep heapin 
            self.__heapify_down(key)
        print(self.min_heap)



'''
Problem 2
Dijkstras minimum path from s to t
'''
ipq = IndexedPriorityQueue()

def Dijkstras(G, s, t):
    costs = {}
    parents = {}
    visited = []

    for x in G.nodes:
        costs[x] = float('inf')
        parents[x] = None
        ipq.push(x, float('inf')) 
    costs[s] = 0
    ipq.decrease_key(s, 0)

    while len(ipq.min_heap) != 0:
        v = ipq.popmin()
        if v == t:
            break
        for x in G.neighbors(v):
            if x not in visited: 
                relax(G, v, x, costs, parents)
        visited.append(v)

    if parents[t] is None:
        return []
    
    shortestPath = []
    while t is not None:
        shortestPath.append(t)
        t = parents[t]
    shortestPath.reverse()
    return shortestPath

def relax(G, v, x, costs, parents):
    weight = G[v][x]['weight']
    if costs[v] + weight < costs[x]:
        #update total cost and keep track of parents as we go so we can print it out later
        costs[x] = costs[v] + weight
        parents[x] = v
        ipq.decrease_key(x, costs[x])


# make graph and run functions

G = nx.Graph()
G.add_nodes_from([x for x in "abcdef"])
G.add_edge("a","b", weight=14)
G.add_edge("a","c", weight=9)
G.add_edge("a","d", weight=7)
G.add_edge("b","c", weight=2)
G.add_edge("b","e", weight=9)
G.add_edge("c","d", weight=10)
G.add_edge("c","f", weight=11)
G.add_edge("d","f", weight=15)
G.add_edge("e","f", weight=6)
print(Dijkstras(G, "a", "e"))
