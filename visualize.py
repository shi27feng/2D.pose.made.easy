from models import ResNet_Spec, ResNet
import hiddenlayer as hl
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)

model = ResNet(ResNet_Spec[18])
hl_graph = hl.build_graph(model, torch.zeros([1, 3, 512, 512]).to(device=device))
hl_graph.theme = hl.graph.THEMES["blue"].copy()
hl_graph.save('pose_net.png', 'png')
