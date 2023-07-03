import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
MODE = "SWARM"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('neptune_data.csv')
path = 'data/graphs/hyperparam_comparison/'
fields = ['Runtime', 'Counters', 'Iterations']
sns.set_style("whitegrid")
for field in fields:
    if MODE == "BOX":
        sns.boxplot(data=df, x=field, y="parameters/last_activation")
    elif MODE == 'VIOLIN':
        sns.violinplot(data=df, x=field, y="parameters/last_activation", cut=0, bw=.15)
    elif MODE == 'BAR':
        sns.barplot(data=df, x=field, y="parameters/last_activation")
    elif MODE == 'SWARM':
        sns.swarmplot(data=df, x=field, y="parameters/last_activation")
    else:
        raise ValueError(f"Invalid mode: {MODE}")
    plt.savefig(f'{path}{MODE}_{field}.png')
    plt.clf()