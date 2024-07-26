import os

path = "/data/beegfs/home/gosalcds/master_den/data/nba/tensor"

for file in os.listdir(path):
    if file.endswith(".pt"):
        # split by "_" and "."
        parts = file.split("_")
        parts2 = parts[1].split(".")
        print(parts2[0] + "_" 