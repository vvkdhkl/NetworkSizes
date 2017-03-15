# coding: utf-8
#from graph import Graph, Node, Digraph
from collections import defaultdict
from numpy import inf
#from bitstring import BitArray
import time
import sys
import random

class Graph:
     def __init__(self, nodes, adjacencyMat, adjacencyMatUndir, num_edges, num_edges_all):
         self._num_nodes = len(nodes)
         self.nodes = nodes
         self.Time = 0
         self._allSCC = list()
         self.adjMat = adjacencyMat
         self.adjMatUndir = adjacencyMatUndir
         self._num_edges = num_edges
         self._undir_edges = num_edges_all

     @property
     def num_nodes(self):
         return self._num_nodes

     def num_edges(self):
         return self._num_edges

     def getNeighbours(self, node, isDirected):
         #print(self.adjMat[node])
         if (isDirected):
           return (self.adjMat[node] if node in self.adjMat else list())
         else:
           return (self.adjMatUndir[node] if node in self.adjMatUndir else list())

     def sub(self, subnodes):
         return (Graph(subnodes, self.adjMat, self.adjMatUndir, self._num_edges, self._undir_edges))

class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)

     def check(self, item):
        return (item in self.items)

     def clear(self):
        self.items = []

     def getList(self):
        return(self.items)


def bitwiseOR(a, blist):
    c = blist[:]
    #bwor = BitArray(a)
    bwor = a
    #print(blist)
    #print(bwor)
    while (c):
        bwor = bwor | c[0]
        c = c[1:]
    return bwor
    #return (a if blist==[] else bitwiseOR(a|blist[-1],blist[0:-1]))

def concat(list):
    return ('' if list==[] else concat(list[0:-1])+list[-1])

def genBitMask(sz):
    bitMask = 0
    r = random.uniform(0,1)
    for k in range(sz):
      if (r>.5**(k+1)):
        bitMask = 2**(sz-k-1)
        break
    #print(bitMask)
    if (bitMask==0):
      bitMask = 2**sz
    #print(int(bitMask))
    return int(bitMask)
    '''bitMask = ['0']*sz
    r = random.uniform(0,1)
    #print(r)
    for k in range(sz):
      if (r>.5**(k+1)):
        bitMask[sz-1-k] = '1'
        break
    if (k>=sz-1):
        bitMask[0] = '1'
    return( BitArray('0b'+concat(bitMask)))'''


def indest(bitmask,sz):
    #c = int(bitmask.length/sz)
    c = 1
    #print(bitmask)
    bm = list(bin(bitmask))[2:]
    b = list(bm).index('0') if '0' in list(bm) else sz

    #print(b)
    return ((2**b)/0.77351)


def anf(graph):
    #print(graph)
    epsilon = 0.000001
    h=1
    cont = True
    NF = dict()
    NFp = dict()
    indEst = dict()
    dist = dict()
    for node in graph.nodes:
        NF[node] = genBitMask(64)
        #print(NF[node])
    nf = sum([indest(NF[node],64) for node in graph.nodes])

    while (cont):
      nfprev = nf
      NFp = dict(NF)
      '''for node in graph.nodes:
          NF[node] = NFp[node]'''
      for node in graph.nodes:
          NF[node] =  bitwiseOR(NFp[node], ([NFp[y] for y in graph.adjMat[node] if y in graph.nodes]) if ((node in graph.adjMat)) else [])
          #print(node,NF[node])
      for node in graph.nodes:
          indEst[node] = indest(NF[node],64)
          #print(indEst[node])
      nf = sum([indEst[p] for p in indEst])
      print(nf, nfprev)
      if (h>1 and nf<nfprev*(1 + epsilon) and nf>nfprev*(1-epsilon)):
        d = h
        cont = False
      else:
        h += 1
        dist[h] = nf
      #NFp = dict(NF)
    dia = d
    vals = [dist[p] for p in dist]
    avg = sum(vals)/len(vals)
    k=1
    vals.sort()
    #print(dist)
    while (vals[k] < avg):
      k+=1 
      meand = list(dist.keys())[list(dist.values()).index(vals[k])]
    end = max(vals)
    k = 1
    while (vals[k] < .9 * end):
      k+=1
    effd = list(dist.keys())[list(dist.values()).index(vals[k])]
    k = 1
    while (vals[k] < .5 * end):
      k+=1
    
    med = list(dist.keys())[list(dist.values()).index(vals[k])]
    
    return(dia, effd, meand, med)


def bfs(self, subgraph, start, endd, isDirected):
    queue = [[start]]
    #tpath = [start]	
    visited = set()
    #queue.append(start)
    while queue:
        
        path = queue.pop(0)
        lastn = path[-1]
        #print(len(queue), lastn, endd)
        #print(vertex)
        #print(lastn)
        #last_node = path[-1]
        #visited.add(lastn)
        #print(self.get_node(last_node).neighbours)
        if (lastn==endd):
                return(1, len(path))
        if lastn not in visited:
            visited.add(lastn)
            n = self.getNeighbours(lastn, isDirected)
            

            for neighbour in n:
		  
                if (neighbour not in visited) and (neighbour not in queue) and (neighbour in subgraph):
                
                    #new_path = list(path)
                    queue.append(path+[neighbour])
                    #queue.append(new_path)
    return(1, len(path))


#The function to do DFS traversal.
# It uses recursive SCCUtil()
def SCC(self, isDirected):

  # Mark all the vertices as not visited
  # and Initialize parent and visited,
  # and ap(articulation point) arrays
  identified = set()
  stack = []
  index = {}
  boundaries = []

  for v in self.nodes:
    if v not in index:
      to_do = [('VISIT', v)]
      while to_do:
        operation_type, v = to_do.pop()
        if operation_type == 'VISIT':
          index[v] = len(stack)
          stack.append(v)
          boundaries.append(index[v])
          to_do.append(('POSTVISIT', v))
                    # We reverse to keep the search order identical to that of
                    # the recursive code;  the reversal is not necessary for
                    # correctness, and can be omitted.
          to_do.extend(
          reversed([('VISITEDGE', w) for w in self.getNeighbours(v, isDirected)]))
        elif operation_type == 'VISITEDGE':
          if v not in index:
            to_do.append(('VISIT', v))
          elif v not in identified:
            while index[v] < boundaries[-1]:
              boundaries.pop()
        else:
         #operation_type == 'POSTVISIT'
          if boundaries[-1] == index[v]:
            boundaries.pop()
            scc = set(stack[index[v]:])
            del stack[index[v]:]
            identified.update(scc)
            yield scc



def dist_metrics(self, subgraph, directed):
  #my_network = nx.Graph() # creating an empty network object
  #nodes = range(1, 6)
  #edges = [(1,2), (2,3), (2,4), (3,4), (4,5)]
  #my_network.add_nodes_from(nodes)
  #my_network.add_edges_from(edges)

  #number_of_nodes = len(nodes)
  #number_of_edges = len(edges)

  if self.num_edges() >= 1:
    diameter = 1

  distances = []
  distance_sum = 0
  dist_matrix = dict()
  dist_matrix = defaultdict(lambda: inf, dist_matrix)

  for x in subgraph:
    dist_matrix[(x,x)] = 0

  for i in self.adjMat:
    for j in self.adjMat[i] : #i = edge.source.index
      #j = edge.dest.index
      if (i in subgraph and j in subgraph):
        dist_matrix[(i,j)] = 1
        if not (directed):
          dist_matrix[(j,i)] = 1
  #print(sys.getsizeof(dist_matrix))

  #pk = [p.index for p in self._nodes]
  for k in subgraph:
    for i in subgraph:
      for j in subgraph:
        if (not directed):
          dist_matrix[(i,j)] = min(dist_matrix[(i,j)], dist_matrix[(j,i)])
          dist_matrix[(j,i)] = min(dist_matrix[(i,j)], dist_matrix[(j,i)])
        if dist_matrix[(i,j)] > dist_matrix[(i,k)] + dist_matrix[(k,j)]:
            dist_matrix[(i,j)] = dist_matrix[(i,k)] + dist_matrix[(k,j)]
        
            if diameter < dist_matrix[(i,j)]:
                diameter = dist_matrix[(i,j)]
 

  for i in subgraph:
    for j in subgraph:
        if dist_matrix[(i,j)] != inf:
            distances.append(dist_matrix[(i,j)])
            distance_sum = distance_sum + dist_matrix[(i,j)]
  #print([(x,dist_matrix[x]) for x in dist_matrix])
  mean_distance = distance_sum/len(distances)
  distances = sorted(distances)
  median_distance = distances[int(len(distances)/2)]
  effective_distance = distances[int(len(distances)*.9)]

  #print(dist_matrix)
  #print(number_of_paths)
  #print(diameter)
  #print(distances)
  #print(distance_sum)
  return(diameter,mean_distance, median_distance, effective_distance)

def random_sources(self, graph, parameter, isDirected):
    if parameter>len(graph):
        parameter = len(graph)
    random_sources = random.sample(graph,parameter)


    distances = []
    distance_sum = 0
    diameter = 1

    for source in random_sources:
        for node in graph:
            distance = bfs(self, graph, source, node, isDirected)[1]
            #print(bfs(self, graph, source, node))
            distances.append(distance)
            distance_sum = distance_sum + distance
            if diameter < distance:
                diameter = distance

    mean_distance = distance_sum/len(distances)
    distances = sorted(distances)
    median_distance = distances[int(len(distances)/2)]
    effective_diameter = distances[int(len(distances)*0.9)]

    return (median_distance, mean_distance, diameter, effective_diameter)

def random_pairs(self, graph, parameter, isDirected):
    if parameter>len(graph):
        parameter = len(graph)
    #random_pairs = []
    distances = []
    distance_sum = 0
    diameter = 1
    for i in range(parameter):
        random_pair = random.sample(graph,2)
        #print(random_pair)
        #random_pairs.append(random_pair)
		
        #print(pair)
        distance = bfs(self, graph, random_pair[0], random_pair[1], isDirected)[1]
        #print(bfs(graph, pair[0], pair[1]))
        distances.append(distance)
        distance_sum = distance_sum + distance
        if diameter < distance:
            diameter = distance

    

    #print(random_pairs)
    #for pair in random_pairs:
        

    #print(distances)
    mean_distance = distance_sum/len(distances)
    distances = sorted(distances)
    median_distance = distances[int(len(distances)/2)]
    effective_diameter = distances[int(len(distances)*0.9)]

    return(median_distance, mean_distance, diameter, effective_diameter)


def read_graph_in_dimacs_format(f):
    """Reads an undirected graph in DIMACS format from given stream. No error handling!"""
    line = f.readline()
    # Ignore comments at beginning of file
    while line.startswith('#'): # line is a comment
        line = f.readline()
    # Expecting: p edge <nodes> <edges>
    nodes = set()

    adjacency_list = dict() # there's a dummy zero node at the start
    adjacency_list_u = dict()
    #adjacency_list_dir = dict()
    cnt = 1
    ln = line
    while (ln):
        # Expecting: <node1> <node2>
        tokens = ln.split()
        #print(tokens, adjacency_list_u, adjacency_list)
        if (int(tokens[0]) in adjacency_list):
          adjacency_list[int(tokens[0])].add(int(tokens[1]))
          #adjacency_list_u[int(tokens[0])].add(int(tokens[1]))
        else:
          adjacency_list[int(tokens[0])] = set([int(tokens[1])])

        if (int(tokens[0]) in adjacency_list_u):  #adjacency_list_u[int(tokens[0])] = set([int(tokens[1])])
          adjacency_list_u[int(tokens[0])].add(int(tokens[1]))
        else:
          adjacency_list_u[int(tokens[0])] = set([int(tokens[1])])

        if (int(tokens[1]) in adjacency_list_u):
          adjacency_list_u[int(tokens[1])].add(int(tokens[0]))
        else:
          adjacency_list_u[int(tokens[1])] = set([int(tokens[0])])

        nodes.add(int(tokens[0]))
        nodes.add(int(tokens[1]))
        ln = f.readline()
        cnt += 1
        #print(cnt, sys.getsizeof(adjacency_list))
    #print(adjacency_list_u)
    cnt_u = sum([len(adjacency_list_u[n]) for n in adjacency_list_u])
    return Graph(nodes,adjacency_list, adjacency_list_u, cnt, cnt_u)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: longest_path_tree.py <file>")
        sys.exit(1)

    # Read graph (tree) from given file or stdin
    with open(sys.argv[1]) as f:
        graph = read_graph_in_dimacs_format(f)

    #print(graph.nodes)
    #print(graph._allSCC)

    scc = SCC(graph, True)
    wcc = SCC(graph, False)
    lscc = max([x for x in scc], key=len)
    lwcc = max([x for x in wcc], key=len)

    print(len(lscc))
    print(len(lwcc))

    print(time.time())
    print('exact lscc ',dist_metrics(graph, lscc, True))
    print(time.time())
    print('exact lwcc ', dist_metrics(graph, lwcc,False))
    print(time.time())
    print('random pair lscc',random_pairs(graph, lscc, 50, True))
    print(time.time())
    #print('random sources lscc', random_sources(graph, lscc, 10, True))
    #print(time.time())
    print('random pair lwcc', random_pairs(graph, lwcc, 1000, False))
    print(time.time())
    print('random sources lwcc', random_sources(graph, lwcc, 1000, False))
    print(time.time())

    #sumd = 0
    #for p in range(10):
    print(time.time())  
    print(anf(graph.sub(lscc)))
    print(time.time())
    print('run ',p+1,' : diameter = ',dia)
    sumd+=dia
    print('average diameter for lscc = ',sumd/10)

    #sumd = 0
    #for p in range(10):
    print(time.time())
    print(anf(graph.sub(lwcc)))
    print(time.time())
    #print('run ',p+1,' : diameter = ',dia)
    #sumd += dia
    #print('average diameter for lwcc = ',sumd/10)'''
