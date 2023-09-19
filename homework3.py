'''
COMP 5970/6970 Graph Algorithms Homework 4 coding section
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
Hint: if you store a binary tree in a vector with each new element being the final
element in the vector representing the last leaf node at the deepest level, you can
compute the index of the children of the node at position i as 2i+1 and 2i+2
You cannot import Queue or any other package for this problem.
'''
class IndexedPriorityQueue:
    def __init__(self):
        self.min_heap = []
        self.index = {}

    def push(self, key, value):
        # push to end of list, then check is parent val is greater and if so then swap
        #note if index is odd then you are a left child ie do -1 and /2
        self.min_heap.append(value)
        self.index[key] = value 
        if (len(self.min_heap)-1 % 2 != 0):
            #odd operation 
            parent = self.min_heap[len(self.min_heap) - 2 / 2]
        else: 
            #even operation
            parent = self.min_heap[len(self.min_heap) - 3 / 2] 
        if (parent > value): 
            self._heapify_up(key)

    def popmin(self):
        # first remove the root node and replace it with the last element in the vector.
        self.min_heap[0] = self.min_heap[len(self.min_heap)-1]
        self.__heapify_down(self.min_heap[0])
        return 0
    def peek(self):
        # your code here
        return 0
    def decrease_key(self, key, new_value):
        # your code here 
        return 0
    def __heapify_up(self, key):
        #swapping an element with its parent till heap is restored
        # your code here
        return 0
    def __heapify_down(self, key):
        #swapping an element with right child (unless not available, then use left child) till heap is restored
        # your code here
        return 0



'''
Problem 2
Dijkstras minimum path from s to t
You should use the Indexed priority queue from problem 1
'''
def Dijkstras(G, s, t):
    # your code here
    return 0





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
Dijkstras(G, "a", "e")
