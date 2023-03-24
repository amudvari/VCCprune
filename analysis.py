#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#%%
directory = "results/deltas"
for (root, dirs, files) in os.walk(f"./{directory}",topdown=True):
    for f in files:
        print(f)
        df = pd.read_csv(f"{directory}/" + f)
        sns.set_theme()
        df.plot(x="epochs", y="test_accs")
        fname = f.split('_')[4].split('.csv')[0].replace('.', '')
        plt.title(f.split('_')[4].split('.csv')[0])
        plt.savefig(f"{directory}/" + fname + ".png")
#%%
directory = "results/resolution_comp"
for (root, dirs, files) in os.walk(f"./{directory}",topdown=True):
    for f in files:
        if f.endswith('.csv'):
            df = pd.read_csv(f"{directory}/" + f)
            sns.set_theme()
            df.plot(x="epochs", y="test_accs")
            fname = f.split('res_comp')[1].split('.csv')[0].replace('.', '')
            title = f"res_comp{fname}"
            plt.title(title)
            plt.savefig(f"{directory}/" + title + ".png")
# %%
directory = "results/other"
for (root, dirs, files) in os.walk(f"./{directory}",topdown=True):
    for f in files:
        if f.endswith('.csv'):
            print(f)
            df = pd.read_csv(f"{directory}/" + f)
            sns.set_theme()
            df.plot(x="epochs", y="test_accs")
            title="STL10"
            plt.title(title)
            plt.savefig(f"{directory}/" + title + ".png")

# %%
