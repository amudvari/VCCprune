#%%
## Libraries
import deprune
import prune
from utils import plot, plot_with_offset

experiments_list = [
    # "Figure_6",
    # "Figure_7",
    # "Figure_8a",
    # "Figure_8b",
    # "Figure_8c",
    # "Figure_10",
    "Figure_11",
]

################ Figure 6 ##########################
# Execution ~ 3 hours
if "Figure_6" in experiments_list:
    deprune.depruning("CIFAR10", training_epochs=7,
                    prune_1_epochs=15, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                    filename="deprune.csv", lr=1e-2,
                    result_directory="figures/Figure_6")
    deprune.depruning("CIFAR10", training_epochs=22,
                    prune_1_epochs=0, prune_2_epochs=0,
                    prune_1_budget=0, prune_2_budget=0,
                    filename="no_compression.csv", lr=1e-2,
                    result_directory="figures/Figure_6")
    deprune.depruning("CIFAR10", training_epochs=0,
                    prune_1_epochs=22, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                    filename="high_compression.csv", lr=1e-2,
                    result_directory="figures/Figure_6")
    plot(['deprune.csv', 'high_compression.csv', 'no_compression.csv'],
        result_name="Figure_6.png", directory="figures/Figure_6")

################ Figure 7 ##########################
# Execution ~ 30 hours
if "Figure_7" in experiments_list:
    deprune.depruning("Imagenet100", training_epochs=12,
                    prune_1_epochs=25, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                      filename="deprune.csv", lr=1e-3,
                    result_directory = "figures/Figure_7")
    deprune.depruning("Imagenet100", training_epochs=37,
                    prune_1_epochs=0, prune_2_epochs=0,
                    prune_1_budget=0, prune_2_budget=0,
                    filename="no_compression.csv", lr=1e-3,
                    result_directory="figures/Figure_7")
    deprune.depruning("Imagenet100", training_epochs=0,
                    prune_1_epochs=37, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                    filename="high_compression.csv", lr=1e-3,
                    result_directory="figures/Figure_7")
    plot(['deprune.csv', 'high_compression.csv', 'no_compression.csv'],
        result_name="Figure_7.png", directory="figures/Figure_7")

################ Figure 8 ##########################
# Execution ~ 3 hours each resolution compression
if "Figure_8a" in experiments_list:
    # Resolution Compression 1
    deprune.depruning("CIFAR10", training_epochs=7,
                    prune_1_epochs=15, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                    filename="deprune.csv", resolution_comp=1, lr=1e-2,
                    result_directory="figures/Figure_8a")
    deprune.depruning("CIFAR10", training_epochs=22,
                    prune_1_epochs=0, prune_2_epochs=0,
                    prune_1_budget=0, prune_2_budget=0,
                    filename="no_compression.csv", resolution_comp=1, lr=1e-2,
                    result_directory="figures/Figure_8a")
    deprune.depruning("CIFAR10", training_epochs=0,
                    prune_1_epochs=22, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                    filename="high_compression.csv", resolution_comp=1, lr=1e-2,
                    result_directory="figures/Figure_8a")
    plot(['deprune.csv', 'high_compression.csv', 'no_compression.csv'],
        result_name="Figure_8a.png", directory="figures/Figure_8a")

if "Figure_8b" in experiments_list:
    # Resolution Compression 2
    deprune.depruning("CIFAR10", training_epochs=7,
                    prune_1_epochs=15, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                    filename="deprune.csv", resolution_comp=2, lr=1e-2,
                    result_directory="figures/Figure_8b")
    deprune.depruning("CIFAR10", training_epochs=22,
                    prune_1_epochs=0, prune_2_epochs=0,
                    prune_1_budget=0, prune_2_budget=0,
                    filename="no_compression.csv", resolution_comp=2, lr=1e-2,
                    result_directory="figures/Figure_8b")
    deprune.depruning("CIFAR10", training_epochs=0,
                    prune_1_epochs=22, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                    filename="high_compression.csv", resolution_comp=2, lr=1e-2,
                    result_directory="figures/Figure_8b")
    plot(['deprune.csv', 'high_compression.csv', 'no_compression.csv'],
        result_name="Figure_8b.png", directory="figures/Figure_8b")

if "Figure_8c" in experiments_list:
    # Resolution Compression 3
    deprune.depruning("CIFAR10", training_epochs=7,
                    prune_1_epochs=15, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                    filename="deprune.csv", resolution_comp=3, lr=1e-2,
                    result_directory="figures/Figure_8c")
    deprune.depruning("CIFAR10", training_epochs=22,
                    prune_1_epochs=0, prune_2_epochs=0,
                    prune_1_budget=0, prune_2_budget=0,
                    filename="no_compression.csv", resolution_comp=3, lr=1e-2,
                    result_directory="figures/Figure_8c")
    deprune.depruning("CIFAR10", training_epochs=0,
                    prune_1_epochs=22, prune_2_epochs=0,
                    prune_1_budget=4, prune_2_budget=0, delta=0.01,
                    filename="high_compression.csv", resolution_comp=3, lr=1e-2,
                    result_directory="figures/Figure_8c")
    plot(['deprune.csv', 'high_compression.csv', 'no_compression.csv'],
        result_name="Figure_8c.png", directory="figures/Figure_8c")


################ Figure 10 ##########################
# Execution ~ 10 hours
if "Figure_10" in experiments_list:
    prune.pruning("CIFAR10", training_epochs=30,
                prune_1_epochs=10, prune_2_epochs=10,
                prune_1_budget=32, prune_2_budget=4, delta=0.01,
                filename="prune.csv",
                result_directory="figures/Figure_10")
    prune.pruning("CIFAR10", training_epochs=50,
                prune_1_epochs=0, prune_2_epochs=0,
                prune_1_budget=0, prune_2_budget=0, delta=0.01,
                filename="no_compression.csv",
                result_directory="figures/Figure_10")
    prune.pruning("CIFAR10", training_epochs=0,
                prune_1_epochs=50, prune_2_epochs=0,
                prune_1_budget=32, prune_2_budget=0, delta=0.01,
                filename="32Comp.csv",
                result_directory="figures/Figure_10")
    prune.pruning("CIFAR10", training_epochs=0,
                prune_1_epochs=50, prune_2_epochs=0,
                prune_1_budget=4, prune_2_budget=0, delta=0.01,
                filename="4Comp.csv",
                result_directory="figures/Figure_10")

    # Top
    plot(['prune.csv', '4Comp.csv', '32Comp.csv', "no_compression.csv"],
        result_name="Figure_10a.png", directory="figures/Figure_10")

    # Bottom 
    plot_with_offset(
        ['prune.csv', '4Comp.csv', '32Comp.csv', "no_compression.csv"],
        offsets={"no_compression": [0, 30], "4Comp": [40, 50], "32Comp": [30, 40]},
        result_name="Figure_10b.png", directory="figures/Figure_10")

################ Figure 11 ##########################
# Execution ~ 2 hours
if "Figure_11" in experiments_list:
    prune.pruning("STL10", training_epochs=30,
                prune_1_epochs=15, prune_2_epochs=15,
                prune_1_budget=32, prune_2_budget=4, delta=0.01,
                filename="prune.csv", lr=1e-3,
                result_directory="figures/Figure_11")
    prune.pruning("STL10", training_epochs=60,
                prune_1_epochs=0, prune_2_epochs=0,
                prune_1_budget=0, prune_2_budget=0, delta=0.01,
                filename="no_compression.csv", lr=1e-3,
                result_directory="figures/Figure_11")
    prune.pruning("STL10", training_epochs=0,
                prune_1_epochs=60, prune_2_epochs=0,
                prune_1_budget=32, prune_2_budget=0, delta=0.01,
                filename="32Comp.csv", lr=1e-3,
                result_directory="figures/Figure_11")
    prune.pruning("STL10", training_epochs=0,
                prune_1_epochs=60, prune_2_epochs=0,
                prune_1_budget=4, prune_2_budget=0, delta=0.01,
                filename="4Comp.csv", lr=1e-3,
                result_directory="figures/Figure_11")

    # Top
    plot(['prune.csv', '4Comp.csv', '32Comp.csv', "no_compression.csv"],
        result_name="Figure_11a.png", directory="figures/Figure_11")

    # Bottom
    plot_with_offset(
        ['prune.csv', '4Comp.csv', '32Comp.csv', "no_compression.csv"],
        offsets={"no_compression": [0, 30], "4Comp": [45, 60], "32Comp": [30, 45]},
        result_name="Figure_11b.png", directory="figures/Figure_11")
