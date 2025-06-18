import os
import pandas as pd

from rich.console import Console
from rich.table import Table


## SMD Stats
# get all *.txt files in data_dir

# Iterate over all files in the directory
train_data_point_cnt, test_data_point_cnt = 0, 0
num_features = 0

data_dir = "data_dir/SMD/raw/train"
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        file_path = os.path.join(root, filename)
        # You can perform operations with each file here
        with open(file_path, 'r') as file:
            train_data_point_cnt += sum(1 for line in file)

        if num_features == 0:
            num_features = pd.read_csv(file_path).shape[1]

data_dir = "data_dir/SMD/raw/test"
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        file_path = os.path.join(root, filename)
        # You can perform operations with each file here
        with open(file_path, 'r') as file:
            test_data_point_cnt += sum(1 for line in file)

train_swat = pd.read_csv("data_dir/SWaT/raw/train.csv")
test_swat = pd.read_csv("data_dir/SWaT/raw/test.csv")
train_wadi = pd.read_csv("data_dir/WaDi/raw/train.csv")
test_wadi = pd.read_csv("data_dir/WaDi/raw/test.csv")

# reporting
console = Console()
#console.print("[bold magenta]Dataset Sumamries[/bold magenta]")

table_combined = Table(title="[bold magenta]Dataset Sumamries[/bold magenta]")
table_combined.add_column("-", style="cyan", justify="right")
table_combined.add_column("SMD", style="green", justify="right")
table_combined.add_column("SWaT", style="yellow", justify="right")
table_combined.add_column("WaDI", style="blue", justify="right")

# Populate the table with shared features
table_combined.add_row("Num Features", f"{num_features}", f"{train_swat.shape[1]}", f"{train_wadi.shape[1]}")
table_combined.add_row("Num Train Samples", f"{train_data_point_cnt}", f"{train_swat.shape[0]}", f"{train_wadi.shape[0]}")
table_combined.add_row("Num Test Samples", f"{test_data_point_cnt}", f"{test_swat.shape[0]}", f"{test_wadi.shape[0]}")
table_combined.add_row("Test/Train Ratio", f"{test_data_point_cnt / train_data_point_cnt:.3f}",
                       f"{test_swat.shape[0] / train_swat.shape[0]:.3f}", f"{test_wadi.shape[0] / train_wadi.shape[0]:.3f}")

# Print the unified table
console.print(table_combined)
