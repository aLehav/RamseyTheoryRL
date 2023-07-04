# Ramsey Theory RL

## About
Ramsey Theory RL is a project that seeks to use simple RL and a basic graph invariant to pseudo-intelligently search for counterexamples for unknown Ramsey number bounds. The main aims of the project are to either [improve the bounds](https://en.wikipedia.org/wiki/Ramsey%27s_theorem) or to assist in finding more [isomorphic graphs](https://users.cecs.anu.edu.au/~bdm/data/ramsey.html) for known numbers to help future efforts. 

Our algorithm is an informed Best First Search. At each iteration, all neighbors (1 changed edge away) to the current graph are numerically represented by their invariant. The invariant is counts of all 11 posible isomorphic 4-graphs inside of them. The neighbors with invariants not yet expanded are then passed through a heuristic which determines which graph to expand next. Past literature has used the count of 4-paths as a heuristic. We use a DNN with 1 hidden layer that we iteratively train on seen graphs, to encourage straying away from known areas. 

The algorithm can pretrain on known graphs and can start from a random graphs, known counterexamples from a smaller $n$, and known counterexamples from the same $n$. Logging is done through neptune.ai.

Our algorithm has a runtime of O($n^2 * n^{\min({4,s,t})}$) per step. As such, values like R(3,10) are less practical to explore, and so our current focus is on R(4,6) and R(5,5).

## Getting Started
- Sign up with [Neptune AI](https://neptune.ai/)
- Create a project called RamseyRL with a model called RAM-HEUR
- Get your Neptune API token and name
- Follow the steps in  `C:\Users\adaml\OneDrive\SC Files\RamseyTheoryRL\RamseyTheoryRL\ColabRunner.ipynb`

## Future Changes

- Improving documentation and usability
- Removing and rearranging older content
- Integrating pip package to Getting Started portion

# Contributors
<a href="https://github.com/aLehav/RamseyTheoryRL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aLehav/RamseyTheoryRL" />
</a>
