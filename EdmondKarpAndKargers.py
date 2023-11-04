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
import numpy as np
from plotnine import *
from networkx.algorithms import bipartite
import random
import sys 
import concurrent.futures

parser = argparse.ArgumentParser()
args = parser.parse_args() # no arguments but im leaving this here

'''
Problem 1
Implement maximum cardinality matching given a bipartite graph.
Feel free to use either Hopcraft-Karp or Edmonds-Karp
The nodes will have an attribute "bipartite" with values 0 or 1 
Output a picture of the graph with edges in the matching in a different color
Use the bipartite_layout to draw it.
'''
def bfs(G, s, t, parent):
    visited = [s]
    visitedEdges = []
    queue = [s]

    while len(queue) != 0:
        u = queue.pop(0)
        if u == t: 
            cur = t
            while (cur != s): 
                visitedEdges.append(((cur, parent[cur]), G[cur][parent[cur]]))
                cur =  parent[cur]
            return (True, visitedEdges)
        for v in G.neighbors(u):
            if v not in visited and (G[u][v]['capacity']-G[u][v]['flow'] > 0  or G[u][v]['Back Capacity']-G[u][v]['Back Flow'] < 0): 
                #i have realized that this way of doing back capacity with an attribute doesn't always work in practice since you could still move forward in the graph 
                #even when at capacity since the back capacity will be negative, but I have not found a way to resolve this without changing the starter code to use a directional graph somehow
                queue.append(v)
                visited.append(v)
                parent[v] = u
    return (False, visitedEdges)

def maximum_matching(G, s, t):
    visitedEdges = []
    while True:
        parent = {}
        (pathBool, visitedEdges) =  bfs(G, s, t, parent)
        if not pathBool:
            break
        bottleneck_value = min([edge[1]['capacity']-edge[1]["flow"] for edge in visitedEdges])
        for edge in visitedEdges:
            x = edge[0][0]
            y = edge[0][1]
            G[x][y]['flow'] = G[x][y]['flow'] + bottleneck_value
            G[x][y]['Back Flow'] = G[x][y]['Back Flow'] - bottleneck_value
    #iterate over edges and look for those with flow which represent the matching
    matching = []
    nodes = []
    for x,y in G.edges():
        if G[x][y]['flow'] > 0:
            matching.append((x,y))
            nodes.append(x)
            nodes.append(y)
    return matching



'''
Problems 2 and 3
Implement Karger's min-cut algorithm
Note: the input is a multi-graph
On each input graph, run 200 iterations of Kargers, report the minimum cut size, and plot a distribution of cut sizes across iterations
I suggest using plotnine for this, a python implementation of ggplot 
'''
def Kargers(G, num_trials=200):
    def karger_single_trial(graph):
        while len(graph.nodes()) > 2:
            x, y, z = random.choice(list(graph.edges))
            graph = nx.contracted_nodes(graph, x, y, self_loops=False)
        return len(graph.edges)

    print('Starting Kargers Algorithm with 200 iterations')
    cutSizes = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_trial = {executor.submit(karger_single_trial, G.copy()): i for i in range(num_trials)}
        for future in concurrent.futures.as_completed(future_to_trial):
            result = future.result()
            cutSizes.append(result)
            progress_bar(len(cutSizes), num_trials, prefix='Progress:', suffix='Complete', length=50)

    print('The min cut size for this batch was', min(cutSizes))
    return cutSizes




#for fun
def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='\u2588'): #unicode for fill char
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()



# make graph and run functions
bipartite_graph = bipartite.random_graph(12,12,0.2,seed=4) # random seed guaranteed to be random, chosen by fair dice roll https://xkcd.com/221/

bipartite_graph.add_nodes_from(['s', 't'])
#adding start and terminal nodes
for node in bipartite_graph.nodes(data=True): 
    if node[0] == 's' or node[0] == 't':
        continue
    elif node[1]['bipartite'] == 0:
        bipartite_graph.add_edge('s', node[0])
    else:
        bipartite_graph.add_edge(node[0], 't')
#first step of kargers
for x,y in bipartite_graph.edges(): 
    bipartite_graph[x][y]['capacity'] = 1
    bipartite_graph[x][y]['flow'] = 0
    bipartite_graph[x][y]['Back Capacity'] = 0
    bipartite_graph[x][y]['Back Flow'] = 0

pos = nx.bipartite_layout(bipartite_graph, bipartite_graph.nodes())
X, Y = nx.bipartite.sets(bipartite_graph)
pos = dict()
pos.update((node, (1, index)) for index, node in enumerate(X))
pos.update((node, (2, index)) for index, node in enumerate(Y))

#removing the starting and ending nodes in the matching
highlighted_edges = []
for tup in maximum_matching(bipartite_graph, 's', 't'): 
    if tup[0] != 's' and tup[0] != 't' and tup[1] != 's' and tup[1] != 't':
        highlighted_edges.append(tup)
bipartite_graph.remove_node('s')
bipartite_graph.remove_node('t')

nx.draw(bipartite_graph, pos, with_labels=True)
nx.draw_networkx_edges(bipartite_graph, pos, edgelist=highlighted_edges, edge_color='r', width=2)
plt.savefig("MaxiumumMatching.png")

# I make a complete graph and remove a lot of edges

G = nx.complete_graph(100)

random.seed(4)



for (x,y) in G.edges:
    if random.random() > 0.1:
        G.remove_edge(x,y)

G = nx.MultiGraph(G)

# I make a complete graph and remove more edges between two sets of nodes than within those sets
G2 = nx.complete_graph(100)

for (x,y) in G2.edges:
    #print(x,y)
    if (x < 50 and y > 50) or (x > 50 and y < 50):
        if random.random() > 0.05:
            #print("yes")
            G2.remove_edge(x,y)
    else:
        if random.random() > 0.4:
            #print("and_yes")
            G2.remove_edge(x,y)

G2 = nx.MultiGraph(G2)

minCut1 = Kargers(G)
minCut2 = Kargers(G2)


# Use ggplot to create the plot
plot = ggplot() + aes(x=range(1, len(minCut1) + 1), y=minCut1) + geom_line() + labs(x="Iterations", y="Min Cut Size")
plot.save("mincutAverage1.png")

plot = ggplot() + aes(x=range(1, len(minCut2) + 1), y=minCut2) + geom_line() + labs(x="Iterations", y="Min Cut Size")
plot.save("mincutAverage2.png")
