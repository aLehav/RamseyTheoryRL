import sys
import igraph as ig
import timeit
sys.path.append("..")
from utils.gfeatures import *
from utils.guseful import *
import multiprocessing

# Timing tests
def main():
  totalStartTime = timeit.default_timer()

  # Useful Tests -------------------------------------------
  # Test for is_complete
  G1 = ig.Graph.Full(5)
  assert is_complete(G1) == True

  G2 = ig.Graph([(0, 1), (0, 2), (1, 2), (2, 3)])  # Not a complete graph
  assert is_complete(G2) == False

  # Test for has_independent_set_of_size_k
  G3 = ig.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
  assert has_independent_set_of_size_k(G3, 2) == True

  G4 = ig.Graph.Full(4)  # Complete graph, no independent set of size 2
  assert has_independent_set_of_size_k(G4, 2) == False

  # Test for has_kn
  G5 = ig.Graph.Full(3)  # Subgraph of size 3 is a complete graph
  assert has_kn(G5, 3) == True

  G6 = ig.Graph([(0, 1), (1, 2)])  # Path of length 2, no K3 subgraph
  assert has_kn(G6, 3) == False
  print('Useful Tests passed')

  # Feature Tests ----------------------------------------------
  # Test 1: Complete graph with 4 nodes
  G1 = ig.Graph.Full(4)
  assert count_subgraph_structures(G1) == {"K_4": 1, "K_{1,3}": 0, "P_4": 0, "C_4": 0,
                                          "K_3+e": 0, "K_4-e": 0, "K_3": 0, "P_3": 0, "K_2": 0, "2K_2": 0, "E_4": 0}
  print("Test 1 passed")

  # Test 2: Path graph with 4 nodes
  G2 = ig.Graph(n=4, edges=[(i, i+1) for i in range(3)])
  assert count_subgraph_structures(G2) == {"K_4": 0, "K_{1,3}": 0, "P_4": 1, "C_4": 0,
                                          "K_3+e": 0, "K_4-e": 0, "K_3": 0, "P_3": 0, "K_2": 0, "2K_2": 0, "E_4": 0}
  print("Test 2 passed")

  # Test 3: Complete bipartite graph K_{1,3}
  G3 = ig.Graph.Full_Bipartite(1, 3)
  assert count_subgraph_structures(G3) == {"K_4": 0, "K_{1,3}": 1, "P_4": 0, "C_4": 0,
                                          "K_3+e": 0, "K_4-e": 0, "K_3": 0, "P_3": 0, "K_2": 0, "2K_2": 0, "E_4": 0}
  print("Test 3 passed")

  # Test 4: Update feature from edge in a complete graph
  G1.delete_edges(G1.get_eid(0, 1))
  features = count_subgraph_structures(G1)
  update_feature_from_edge(G1, 0, 1, features)
  assert features == {"K_4": 1, "K_{1,3}": 0, "P_4": 0, "C_4": 0, "K_3+e": 0,
                      "K_4-e": 0, "K_3": 0, "P_3": 0, "K_2": 0, "2K_2": 0, "E_4": 0}
  print("Test 4 passed")

  # Test 5: Update feature from edge in a path graph
  features = count_subgraph_structures(G2)
  update_feature_from_edge(G2, 0, 2, features)
  assert features == {"K_4": 0, "K_{1,3}": 0, "P_4": 0, "C_4": 0, "K_3+e": 1,
                      "K_4-e": 0, "K_3": 0, "P_3": 0, "K_2": 0, "2K_2": 0, "E_4": 0}
  print("Test 5 passed")


  print("Count_subgraph_structures Tests-----------------------------------------")
  G9 = ig.Graph.Full(35)
  startTime = timeit.default_timer()
  count_subgraph_structures(G9)
  print(f"Single Threaded Time Elapsed {timeit.default_timer() - startTime}")

  startTime = timeit.default_timer()
  count_subgraph_structures_parallel(G9)
  print(f"Multi Threaded Time Elapsed {timeit.default_timer() - startTime}\n")

  # Timing tests
  print("Count_subgraphs_from_edge Tests")
  startTime = timeit.default_timer()
  count_subgraphs_from_edge(G9, 3,4)
  print(f"Single Threaded Time Elapsed {timeit.default_timer() - startTime}")

  startTime = timeit.default_timer()
  count_subgraphs_from_edge_parallel(G9,3,4)
  print(f"Multi Threaded Time Elapsed {timeit.default_timer() - startTime}")
  print(f"Total Time elapsed: {timeit.default_timer() - totalStartTime}")

# def add_one(n):
#   my_temp = list(range(100))
#   my_temp[3] = 3
#   for j in range(100):
#     my_temp[j] = my_temp[j] > 5
#   return n+1

# def main():
#   numbers = list(range(10000000))
#   startTime = timeit.default_timer()
#   temp_numbers = list(range(10000000))
#   for i in range(len(temp_numbers)):
#     temp_numbers[i] = add_one(temp_numbers[i])
#   print(f"Single Threaded Time Elapsed {timeit.default_timer() - startTime}")

#   startTime = timeit.default_timer()
#   with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#       results = pool.map(add_one, numbers)
#   print(f"Multi Threaded Time Elapsed {timeit.default_timer() - startTime}")



if __name__ == '__main__':
  main()