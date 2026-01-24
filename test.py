import torch
import torch.nn as nn

# linear = nn.Linear(1,80)
# x = nn.Parameter(torch.zeros((5,1)))
# print(linear(x).unsqueeze(0).shape)

# def action_mask(seq_len,action_len):
#     mask = torch.ones((seq_len,seq_len),dtype=torch.bool)
#     action_mask = torch.tril(torch.ones((action_len,action_len)),diagonal=0)
#     mask[:,-action_len:] = False
#     mask[-action_len:,-action_len:] = action_mask
#     return mask
# attention = torch.randn(10,3,128,128)
# mask = action_mask(128,5)
# attention = attention.masked_fill(~mask,-1e9)
# print(attention)
import json
for i in range(4,9):
    print(i)
    map_name = "/home/mapf-gpt/temp/medium-mazes-seed-001"+str(i)+".json"
    with open(map_name, "r") as f:
        data = json.load(f)