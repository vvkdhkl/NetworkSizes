Coursework project: Network statistics calculation

Large Networks: ~million nodes

Algorithms implemented: Random sources/pairs, Warshall-Floyd, ANF

Problem: Read network files and compute the sizes: mean, median, effective and exact diameters.

For the statistics computations, we first implemented exact distance calculation algorithm (Warshall-Floyd), which worked well with small graphs when we tested. However, on testing it with larger graphs, it was very bad with the runtime, such that we couldn’t get it finished for large graphs beyond Wiki-Vote lscc. The program was always killed midway. O(n^3) indeed proved to be very heavy cost.
Then we focused on getting ANF running. The algorithm was implemented based on the paper on ANF (http://dl.acm.org/citation.cfm?doid=775047.775059) , using Flajolet-Martin (FM) bitmask counters. The implementation is discussed below:

Hashing: A simple probabilistic (uniformly random bit flips) hashing was used. Each node was mapped to a 64-bitstring where position of exactly one ith bit was 1 with the probability .5i+1  and all other 0 bits. This worked because of the relatively large bitsize of the bitmask. It was then mapped to an integer counter as explained below.
FM based BitMask (counter): We tried with various sizes of counters, but in the end settled for 64 bit counter as it gave good results. We tried with two implementations:
	Bitarrray based counter: this counter was implemented as a BitArray of 64 bits. But this proved to be very inefficient on profiling using cProfile. Most of the computation was in the bitwise OR operation in this data structure. Thus, we needed to come up with something better.
	Integer bitmasks: It turns out that python native 64 bit integers are both very fast (in the order of one of a hundredth) and also enough for our purpose, so we implemented the bit mask as an integer and then performed all bitwise OR computations and Individual Estimate (the indicator of neighbourhood size for a node) natively, resulting in fast performance.

epsilon is the limit of ratio between difference between two consecutive Neighborhood function values with the present value. (stop when Ng(h)>(1-epsilon)*Ng(h-1) and Ng(h)<(1+epsilon)*Ng(h-1)
Individual Estimate (of Ng(h, node)) : This was implemented as-is in the paper referred. We could get nice convergent values which were then used to calculate not just the diameter but also the effective diameter. Even using a small epsilon (convergence limit, +/- 0.000001 in proportion) gavegood approximates, which was possible because of dense networks. The total Neighbourhood function was then the sum of the individual estimates of all nodes.
Then we implemented a non-recursive bfs, random sources and random pairs algorithms. BFS did not work as well as we expected, for large graphs, although we tested it with small ones and it was working fine. Because of BFS, many runs of random sources and random pairs algorithms did not terminate for large graphs for 1000 or more random sources and pairs. It gave results for a small number of BFS runs, eg we tried with 50, but it wouldn’t then give good approximates for the network statistics.
We tried to generate a graph with multiple epsilon parameters for the largest graph but failed because we could not run it enough times to produce graph (program ran for multiple hours each time, loading the graph was very expensive). But we experimented with epsilon values 0.05, 0.02, 0.01, 0.001, 0.000001 and we found it gave good approximates for epsilon <=0.02.
Here epsilon is the limit of ratio between difference between two consecutive Neighborhood function values with the present value. (stop when Ng(h)>(1-epsilon)*Ng(h-1) and Ng(h)<(1+epsilon)*Ng(h-1)
