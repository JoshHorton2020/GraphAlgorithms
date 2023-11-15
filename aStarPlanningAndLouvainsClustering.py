import argparse
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
from networkx.algorithms import bipartite
import queue

parser = argparse.ArgumentParser()
args = parser.parse_args() # no arguments but im leaving this here

'''
Implementing Dijkstras min path and A* search on the maze graph G to find the best
route from s to t. nodes are named with their coordinate position. 
Using manhattan distance as it is on a grid graph as the heuristic
'''

def Dijkstra(G, s, t):
    costs = {}
    parents = {}
    visited = []
    ipq = queue.PriorityQueue()

    total_touched_djik = 0
    total_popped_djik = 0
    for x in G.nodes:
        costs[x] = float('inf')
        parents[x] = None
        ipq.put((float('inf'), x))
        total_touched_djik += 1
    costs[s] = 0
    ipq.put((0, s))

    while not ipq.empty():
        priority, v = ipq.get()
        total_popped_djik += 1
        if v == t:
            break
        for x in G.neighbors(v):
            if x not in visited:
                relax(G, v, x, costs, parents, ipq, total_touched_djik)
        visited.append(v)

    if parents[t] is None:
        return []

    shortestPath = []
    while t is not None:
        shortestPath.append(t)
        t = parents[t]
    shortestPath.reverse()
    return shortestPath, total_touched_djik, total_popped_djik


def relax(G, v, x, costs, parents, ipq, ttd):
    weight = G[v][x]['weight']
    if costs[v] + weight < costs[x]:
        # update total cost and keep track of parents as we go so we can print it out later
        costs[x] = costs[v] + weight
        parents[x] = v
        ipq.put((costs[x], x))
        ttd += 1


def Astar(G, s, t):
    costs = {}
    parents = {}
    visited = []
    ipq = queue.PriorityQueue()
    total_touched_astar = 0 
    total_popped_astar = 0  

    for x in G.nodes:
        costs[x] = float('inf')
        parents[x] = None
        ipq.put((float('inf'), x))
        total_touched_astar += 1
    costs[s] = 0
    ipq.put((0, s))
    total_touched_astar += 1
    while not ipq.empty():
        priority, v = ipq.get()
        total_popped_astar += 1
        if v == t:
            break
        for x in G.neighbors(v):
            if x not in visited:
                a_star_relax(G, v, x, costs, parents, ipq, t, total_touched_astar)
        visited.append(v)

    if parents[t] is None:
        return []

    shortestPath = []
    while t is not None:
        shortestPath.append(t)
        t = parents[t]
    shortestPath.reverse()
    return shortestPath, total_touched_astar, total_popped_astar


def a_star_relax(G, v, x, costs, parents, ipq, t, tta):
    weight = G[v][x]['weight']
    if costs[v] + weight < costs[x]:
        costs[x] = costs[v] + weight + heuristic(x, t)
        parents[x] = v
        ipq.put((costs[x], x))
        tta += 1
#manhattan distance
def heuristic(a, b): 
    x1, y1 = a
    x2, y2 = b
    #simply difference of x's and y's added together gives total distance
    return abs(x1 - x2) + abs(y1 - y2)





'''
Implementing the louvain method for community detection on the Graph G and visualizing the final graph colored by cluster
'''
def louvain(G, depth):
    depthCheck = 0
    dendrogram = []
    community = {}
    community_list = []
    for i,v in enumerate(G.nodes): 
        community[v] = i 
        community_list.append([v])
    any_change = True
    while any_change: 
        while any_change: 
            any_change = False
            touched = []
            for v in G.nodes(): 
                if v in touched: 
                    continue
                best_communnity = None 
                best_delta = 0 
                first = True
                for x in G.neighbors(v): 
                    targetCommunity = community[x]
                    if v in community_list[targetCommunity]:
                        continue
                    #sum num edges in each community and subtract expected value which is E (edge i, j)= (degree of i * degree j over 2 * num edges in G) 
                    if first: 
                        original = community[v]
                        first = False
                    
                    sharedEdgeCount = 0
                    edgeList = []
                    community_list[targetCommunity].append(v)
                    for node in community_list[targetCommunity]: 
                        edgeList.extend(list(G.edges(node)))
                    #remove dupes, geeks for geeks approved pythonic method lol
                    edgeList = list(set(edgeList))

                    secondPartNumerator = 0
                    considered = []
                    for edge in edgeList: 
                        #making sure edge in community and not double counting backwards edges since this is unweighted and undirectional
                        if edge[0] not in community_list[targetCommunity] or edge[1] not in community_list[targetCommunity] or edge in considered or (edge[1], edge[0]) in considered:
                            continue 
                        if v == edge[0] or v == edge[1]:
                            sharedEdgeCount += 1
                        secondPartNumerator += G.degree(edge[0]) * G.degree(edge[1])
                        considered.append(edge)
                    #remove v from community since we are only testing if it is a good fit and not actually moving it yet 
                    community_list[targetCommunity].remove(v)
                    
                    delta = ((sharedEdgeCount) - (secondPartNumerator / (2*G.number_of_edges()))) / (2 * G.number_of_edges())

                    if delta > best_delta: 
                        best_delta = delta 
                        best_communnity = community[x]

                if best_communnity != None: 
                    any_change = True 
                    community[v] = best_communnity 
                    community_list[original].remove(v)
                    community_list[best_communnity].append(v)
                    touched.extend(community_list[best_communnity])
        dendroItem = [item for item in community_list if len(item) > 0]  
        dendrogram.append(dendroItem)
        gPrime = G.copy() #collapse communities into single nodes and make multi graph
        if depth == depthCheck:
            break
        else: 
            depthCheck += 1
        for i, com in enumerate(community_list):
            if len(com) == 0: 
                continue 
            primeNode = com[0]
            toberemoved = []
            for node in com: 
                if node != primeNode: 
                    gPrime = nx.contracted_nodes(gPrime, primeNode, node, self_loops=False)
                    toberemoved.append(node)
            for x in toberemoved: 
                com.remove(x)
        if G != gPrime: 
            any_change = True
            G = gPrime
    
    finalCommunities = dendrogram[0]
    color_map = {}
    for i, sublist in enumerate(finalCommunities):
        for n in sublist:
            color_map[n] = i
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=[color_map[color] for color in G.nodes()], cmap=plt.cm.get_cmap('rainbow'))
    plt.savefig("clusteredGraph.png")
    return dendrogram

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


path, ttd, tpd = Dijkstra(G, (0,0), (0,7))
print('Djikstras Path = ', path)
print(f"Djikstras touched this number of nodes: {ttd} and popped this number of nodes: {tpd}")
apath, tta, tpa = Astar(G, (0,0), (0,7))
print(f'Astar path = ', apath)
print(f"Astar touched this number of nodes: {tta} and popped this number of nodes: {tpa}")

G = nx.Graph()
G.add_nodes_from([x for x in "abcdefghijklmno"])
G.add_edges_from([("a","b"),("a","c"),("a","d"),("b","c"),("b","d"),("c","e"),("d","e")])
G.add_edges_from([("f","g"),("f","h"),("f","i"),("f","j"),("g","j"),("g","h"),("h","i"),("h","j"),("i","j")])
G.add_edges_from([("k","l"),("k","n"),("k","m"),("l","n"),("n","m"),("n","o"),("m","o")])
G.add_edges_from([("e","f"),("j","l"),("j","n")])

print('The final clustering is - ', louvain(G, 0)) 
#added extra param to specify the amount of collapses you want to perform aka the depth of the dendrogram you want returned
#and since we want the clustering containing all the original nodes then we want a depth of 0 since we have no collapsed communities
#but trying any depth other than 0 results in an infinite loop making me think my delta calculation may still be slightly wrong 
#since we never reach any optima and nodes just keep bouncing between communities undecided 
