import torch
model = torch.load('teste.pt')
for key in model.keys():
  print(key, model[key].shape)

