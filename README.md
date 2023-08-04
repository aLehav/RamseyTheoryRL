# Ramsey Theory RL

## About
Ramsey Theory RL is a project that seeks to use simple RL and a basic graph invariant to pseudo-intelligently search for counterexamples for unknown Ramsey number bounds. The main aims of the project are to either [improve the bounds](https://en.wikipedia.org/wiki/Ramsey%27s_theorem) or to assist in finding more [isomorphic graphs](https://users.cecs.anu.edu.au/~bdm/data/ramsey.html) for known numbers to help future efforts. 

Our algorithm is an informed Best First Search. At each iteration, all neighbors (1 changed edge away) to the current graph are numerically represented by their invariant. The invariant is counts of all 11 possible isomorphic 4-graphs inside of them. The neighbors with invariants not yet expanded are then passed through a heuristic which determines which graph to expand next. Past literature has used the count of 4-paths as a heuristic. We use a DNN with 1 hidden layer that we iteratively train on seen graphs, to encourage straying away from known areas. 

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

  def project_fetcher():
      return f"{os.environ.get('NEPTUNE_NAME')}/RamseyRL"

  runner = NeptuneRunner(n=36, s=4, t=6, project=project_fetcher())
  
  new_params = runner.PARAMS
  changed_vals = {'iter_batch':20,
                  'iter_batches':50,
                  'starting_graph':"FROM_CURRENT",
                  'starting_graph_path':'/data/found_counters/r4_5_24_isograph.g6'
                  }
  new_params.update(changed_vals)
  runner.update_params(new_params)
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

## Full Parameter Description

 - **heurstic_type**: Choose from RANDOM, 4PATH, DNN, SCALED_DNN
 - **iter_batch**: Steps to take before updating model data / weights
 - **iter_batches**: None if no stopping value, else num. of iter_batches
 - **starting_graph**: Choose from RANDOM, FROM_PRIOR, FROM_CURRENT, EMPTY
 - **starting_graph_path**: Path to starting graph
 - **starting_graph_index**: Index of starting graph within starting graph index list
 - **starting_edges**: Add new edges if starting FROM_PRIOR
 - **load_model**: Whether or not to load a past version of the model
 - **profiler**: Enable profiler
 - **model**: If passed, when using SCALED_DNN or DNN heuristic, the default model of a DNN with hidden layers of size 32, 16, and then 1 with optimizer=params['optimizer'], loss=params['loss'], metrics=['accuracy'] can be overridden for any tf model that has already had `model.compile` called
 - **training_epochs**: If pretraining on data, the number of epochs to train for
 - **pretrain**: Whether or not pretraining occurs
 - **pretrain_data**: A list of preprocessed csv's to be trained on
 - **epochs**: Epochs to update the model for after each iteration of training
 - **batch_size**: Batch size for training
 - **loss_info**: Info regarding loss used, improves logging readability
 - **alpha**: Learning rate for DNN (i.e. if a counter was found k steps ago, a point that isn't a counter in this iteration would have a value of $\alpha^k$)

## Future Changes

- Improving documentation and usability
- Removing and rearranging older content
- Integrating pip package to Getting Started portion

# Contributors
<a href="https://github.com/aLehav/RamseyTheoryRL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aLehav/RamseyTheoryRL" />
</a>
