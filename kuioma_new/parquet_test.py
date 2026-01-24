import pandas as pd
import numpy as np
import json

parquet_path = "/home/mapf-gpt/LaCAM_data/dataset_configs/10-medium-mazes/temp/LaCAM.parquet"
df_read = pd.read_parquet(parquet_path,engine="pyarrow")
episode_map = json.loads(df_read["map"])
print(df_read[""])
