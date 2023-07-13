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
- Run the following
  
  ### Installing Packages
  ```bash
  pip install RamseyTheoryRL
  pip install -r https://raw.githubusercontent.com/aLehav/RamseyTheoryRL/main/RamseyTheoryRL/requirements.txt --quiet
  ```
  ### Setting Environment Variables
  ```python
  import os
  # Change these for your real token and username
  os.environ['NEPTUNE_API_TOKEN'] = 's0me!l0nGn3pTunEt0k3N='
  os.environ['NEPTUNE_NAME'] = 'yourname'
  ```
  ### Setting Parameters and Path
  ```python
  from RamseyTheoryRL.src.ramsey_checker.test import NeptuneRunner
  import tensorflow as tf

  def setup(runner):
      PARAMS = {'heuristic_type': "SCALED_DNN",  # Choose from RANDOM, 4PATH, DNN, SCALED_DNN
                'iter_batch': 20,  # Steps to take before updating model data / weights
                'iter_batches': 50,  # None if no stopping value, else num. of iter_batches
                'starting_graph': "FROM_PRIOR"}  # Choose from RANDOM, FROM_PRIOR, FROM_CURRENT, EMPTY
      if PARAMS['heuristic_type'] in ["DNN", "SCALED_DNN"]:
          DNN_PARAMS = {'training_epochs': 5, 'epochs': 1, 'batch_size': 32, 'optimizer': 'adam', 'loss': tf.keras.losses.BinaryCrossentropy(
              from_logits=False, label_smoothing=0.2), 'loss_info': 'BinaryCrossentropy(from_logits=False, label_smoothing=0.2)', 'last_activation': 'sigmoid', 'pretrain': True}
          PARAMS.update(DNN_PARAMS)
          if PARAMS['pretrain']:
              CSV_LIST = ['all_leq6', 'ramsey_3_4', 'ramsey_3_5',
                          'ramsey_3_6', 'ramsey_3_7', 'ramsey_3_9']
              PARAMS.update({'pretrain_data': CSV_LIST})
      if PARAMS['starting_graph'] in ["FROM_PRIOR", "FROM_CURRENT"]:
          STARTING_GRAPH_PARAMS = {'starting_graph_path': '/data/found_counters/r4_6_35_isograph.g6',  # Mac: Absolute path
                                      'starting_graph_index': 1  # 0 is default
                                      }
          PARAMS.update(STARTING_GRAPH_PARAMS)
      runner.update_params(PARAMS)

  def project_fetcher():
      return f"{os.environ.get('NEPTUNE_NAME')}/RamseyRL"

  runner = NeptuneRunner(n=36, s=4, t=6, project=project_fetcher())
  setup(runner)
  ```
  ### Running
  ```python
  runner.run()
  ```
  ### (Optional) Running for all Prior Graphs
  ```python
  # Getting max index of starting graph
  # Only to be used when PARAMS['starting_graph] in ["FROM_PRIOR", "FROM_CURRENT"]
  import sys
  import networkx as nx

  def get_max_starting_index(runner):
    counters = nx.read_graph6(sys.path[-1] + runner.PARAMS['starting_graph_path'])
    counters = [counters] if type(counters) != list else counters
    return len(counters)

  for i in range(get_max_starting_index(runner)):
    runner.PARAMS['starting_graph_index'] = i
    runner.run()
  ```

## Future Changes

- Improving documentation and usability
- Removing and rearranging older content
- Integrating pip package to Getting Started portion

# Contributors
<a href="https://github.com/aLehav/RamseyTheoryRL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aLehav/RamseyTheoryRL" />
</a>
