import torch
import torch.nn.functional as F


def MSE_SAM_loss(output, target, alpha=1., beta=1.):
    eps = torch.tensor(1e-30)
    mse = torch.nn.MSELoss(reduction='mean')(output, target)
    # sam = torch.acos((torch.sum(output * target, dim=-1)) / (
    #         torch.norm(output, p=2, dim=-1) * torch.norm(target, p=2, dim=-1)+eps))
    # sam = torch.mean(torch.rad2deg(sam))
    # cos_distance =
    # sam = torch.mean(sam)
    # print('mse:', mse.item(), 'sam:', sam.item())
    # if torch.isnan(mse) or torch.isnan(sam):
    #     print('NAN')
    cos_sim = 1 - torch.nn.CosineSimilarity(dim=-1)(output, target)
    cos_sim = torch.mean(cos_sim)
    # print('mse:', mse.item(), 'sam:', cos_sim.item())
    return alpha * mse  +  beta * cos_sim

#
# a = torch.tensor([1,0,0.,0.,0.,0.,0.])
# b = torch.tensor([0,1.,1.,1.,1,1.,1])
# # #
# loss = MSE_SAM_loss(a, b)
#
# print(loss)
