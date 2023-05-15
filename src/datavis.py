import networkx as nx
import matplotlib.pyplot as plt
import os

def remove_outputs(dir_name):
  test = os.listdir(dir_name)

  for item in test:
    if item.endswith(".pdf"):
      os.remove(os.path.join(dir_name, item))

# output file format: data/name/name_{# vertices}_{idx}.pdf where idx <= # graph counterexamples for that vertex count
def visualize_data(k, l, count):
  name = 'r' + str(k) + str(l)
  for i in range(1, count+1):
    filename = '../data/' + name + '/' + name + '_' + str(i) + '.g6'
    # .g6 files contain multiple graphs
    if os.path.exists(filename):
      graphs = nx.read_graph6(filename)
      # Bug: graphs is type networkx.graph if there is 1 graph, else its a list
      if (type(graphs) != list):
        nx.draw(graphs)
        output_filename = f"{filename[:-3]}_{1}.pdf"
        plt.savefig(output_filename)
        plt.clf()  # Clear the plot for the next graph
      else:
        for idx, g in enumerate(graphs, start=1):
          nx.draw(g)
          output_filename = f"{filename[:-3]}_{idx}.pdf"
          plt.savefig(output_filename)
          plt.clf()  # Clear the plot for the next graph

def main():
  # remove_outputs('../data/r34')
  # remove_outputs('../data/r35')
  # remove_outputs('../data/r36')
  visualize_data(3,4,8)
  visualize_data(3,5,13)
  visualize_data(3,6,17)

if __name__ == '__main__':
  main()