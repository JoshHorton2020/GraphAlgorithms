'''
COMP 5970/6970 Graph Algorithms Homework 7 coding section
requires networkx, argparse
requires python 3.6+ (can get with anaconda or elsewhere, note standard python with mac is python 2)
pip install networkx
pip install argparse
'''

import argparse
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
from networkx.algorithms import bipartite
import random
import queue

parser = argparse.ArgumentParser()
args = parser.parse_args() # no arguments but im leaving this here

'''
Problem 1
Implement Dijkstras min path and A* search on the maze graph G to find the best
route from s to t. nodes are named with their coordinate position. 
Feel free to use the queue package
Report the number of nodes expanded (popped from the queue) as well
as touched (added to the queue)

use euclidean (or manhattan distance as it is on a grid graph) distance to t as your heuristic
'''
def Dijkstra(G, s, t):
    costs = {}
    parents = {}
    visited = []
    ipq = queue.PriorityQueue()

    for x in G.nodes:
        costs[x] = float('inf')
        parents[x] = None
        ipq.put((float('inf'), x))
    costs[s] = 0
    ipq.put((0, s))

    while not ipq.empty():
        priority, v = ipq.get()
        if v == t:
            break
        for x in G.neighbors(v):
            if x not in visited:
                relax(G, v, x, costs, parents, ipq)
        visited.append(v)

    if parents[t] is None:
        return []

    shortestPath = []
    while t is not None:
        shortestPath.append(t)
        t = parents[t]
    shortestPath.reverse()
    return shortestPath


def relax(G, v, x, costs, parents, ipq):
    weight = G[v][x]['weight']
    if costs[v] + weight < costs[x]:
        # update total cost and keep track of parents as we go so we can print it out later
        costs[x] = costs[v] + weight
        parents[x] = v
        ipq.put((costs[x], x))

def Astar(G, s, t):
    costs = {}
    parents = {}
    visited = []
    ipq = queue.PriorityQueue()

    for x in G.nodes:
        costs[x] = float('inf')
        parents[x] = None
        ipq.put((float('inf'), x))
    costs[s] = 0
    ipq.put((0, s))

    while not ipq.empty():
        priority, v = ipq.get()
        if v == t:
            break
        for x in G.neighbors(v):
            if x not in visited:
                a_star_relax(G, v, x, costs, parents, ipq, t)
        visited.append(v)

    if parents[t] is None:
        return []

    shortestPath = []
    while t is not None:
        shortestPath.append(t)
        t = parents[t]
    shortestPath.reverse()
    return shortestPath


def a_star_relax(G, v, x, costs, parents, ipq, t):
    weight = G[v][x]['weight']
    if costs[v] + weight < costs[x]:
        costs[x] = costs[v] + weight + heuristic(x, t)
        parents[x] = v
        ipq.put((costs[x], x))
#manhattan distance
def heuristic(a, b): 
    x1, y1 = a
    x2, y2 = b
    return abs(x1 - x2) + abs(y1 - y2)





'''
Problem 2 Implement the louvain method for community detection on the Graph G. 
visualize the final graph colored by cluster

'''
def louvain(G):
    community = {}
    listy = []
    for i,v in enumerate(G.nodes): 
        community[v] = i 
        listy.append([v])
    any_change = True
    while any_change: 
        while any_change: 
            any_change = False
            for v in G.nodes(): 
                best_communnity = None 
                best_delta = 0 
                for x in G.neighbors(v): 
                    #sum num edges in each community and subtract expected value which is E (edge i, j)= (degree of i * degree j over 2 * num edges in G) 

                    #move v to x
                    targetCommunity = community[x]
                    oldCommunity = community[v]
                    listy[targetCommunity].append(v)

                    #now add all edges possible in community
                    total = 0
                    modularity = 0
                    for node in listy[targetCommunity]: 
                        edgeList = list(G.edges(node))
                    
                    #removing dupes from edge list and checking if they are actually in community
                    edgeList = set(edgeList)

                    for edge in edgeList: 
                        #if edge not in comm do nothing
                        if edge[0] not in listy[targetCommunity] or edge[1] not in listy[targetCommunity]:
                            continue
                        #else add to calculation 
                        else: 
                            total += 1
                            modularity += ( (G.degree[edge[0]] - G.degree[edge[1]]) / (2*G.number_of_edges()) )

                    delta = total-modularity #make equal to modularity for moving v to x.community 

                    if delta > best_delta: 
                        best_delta = delta 
                        best_communnity = community[x]
                if best_communnity != None: 
                    any_change = True 
                    community[v] = best_communnity 
    
    gPrime = G.copy() #MAKE - collapse communities into single nodes and make multi graph
    
    if G != gPrime: 
        any_change = True
        G = gPrime







# make graph and run functions
G = nx.grid_2d_graph(5,8)
G.remove_node((1,1))
G.remove_node((1,2))
G.remove_node((1,3))
G.remove_node((3,1))
G.remove_node((3,3))
G.remove_node((3,4))
G.remove_node((3,5))
G.remove_node((3,6))
G.remove_node((0,5))
G.remove_node((1,5))
nx.set_edge_attributes(G, 1, "weight")

'''
This graph should represent the following maze
_____
t    |
  x |
xx x |
   x |
 x x |
 x   |
 x x |
s    |
-----

'''
print(Dijkstra(G, (0,0), (0,7)))
print(Astar(G, (0,0), (0,7)))


G = nx.Graph()
G.add_nodes_from([x for x in "abcdefghijklmno"])
G.add_edges_from([("a","b"),("a","c"),("a","d"),("b","c"),("b","d"),("c","e"),("d","e")])
G.add_edges_from([("f","g"),("f","h"),("f","i"),("f","j"),("g","j"),("g","h"),("h","i"),("h","j"),("i","j")])
G.add_edges_from([("k","l"),("k","n"),("k","m"),("l","n"),("n","m"),("n","o"),("m","o")])
G.add_edges_from([("e","f"),("j","l"),("j","n")])

louvain(G)
