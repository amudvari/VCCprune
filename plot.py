#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#%%
for (root, dirs, files) in os.walk("./results",topdown=True):
    for f in files:
        print(f)
        df = pd.read_csv("results/" + f)
        sns.set_theme()
        df.plot(x="epochs", y="test_accs")
        fname = f.split('_')[4].split('.csv')[0].replace('.', '')
        plt.title(f.split('_')[4].split('.csv')[0])
        plt.savefig("results/" + fname + ".png")
# %%
