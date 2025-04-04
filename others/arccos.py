import torch
import torch.nn as nn
input1 = torch.tensor([[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[2,7,4.,8,4,7,2,7,4]])/10
input2 = torch.tensor([[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[2,7,4.,8,4,7,2,7,4]])/100

cos =nn.CosineSimilarity(dim=-1)
a = 1.0 - cos(input1,input2)

b = 0