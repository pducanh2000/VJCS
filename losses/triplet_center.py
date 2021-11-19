import torch.nn as nn
import torch
from typing import Tuple
from torch import nn, Tensor
from torch.autograd import Variable 

class TripletCenterLoss(nn.Module):
    def __init__(self, margin=1, num_classes=17):
        super(TripletCenterLoss, self).__init__() 
        self.margin = margin 
        self.ranking_loss = nn.MarginRankingLoss(margin=margin) 
        self.centers = nn.Parameter(torch.randn(num_classes, num_classes)) 
   
    def forward(self, inputs, targets): 
        batch_size = inputs.size(0) 
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1)) 
        centers_batch = self.centers.gather(0, targets_expand) # centers batch 

        # compute pairwise distances between input features and corresponding centers 
        centers_batch_bz = torch.stack([centers_batch]*batch_size) 
        inputs_bz = torch.stack([inputs]*batch_size).transpose(0, 1) 
        dist = torch.sum((centers_batch_bz -inputs_bz)**2, 2).squeeze() 
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability 

        # for each anchor, find the hardest positive and negative 
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], [] 
        for i in range(batch_size): # for each sample, we compute distance 
            dist_ap.append(dist[i][mask[i]].max()) # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i]==0].min()) # mask[i]==0: negative samples of sample i 

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # generate a new label y
        # compute ranking hinge loss 
        y = dist_an.data.new() 
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero 
        loss = self.ranking_loss(dist_an, dist_ap, y)

        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0) # normalize data by batch size 
        return loss, prec    