import math
import torch
from torch import nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=17, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, use_label_smoothing=True):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.to(torch.device('cuda'))
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class AM_Softmax(nn.Module):  # creates the classification layer
    def __init__(self, m=0.35, s=30, feat_dim=2048, num_classes=17, use_gpu=True, epsilon=0.1):
        super(AM_Softmax, self).__init__()
        self.d = feat_dim
        self.m = m
        self.s = s
        self.num_classes = num_classes

        self.weight = torch.nn.Linear(self.d, num_classes, bias=False)
        if use_gpu:
            self.weight = self.weight.cuda()
        bound = 1 / math.sqrt(self.d)
        nn.init.uniform_(self.weight.weight, -bound, bound)
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes, use_gpu=use_gpu)

    def forward(self, x, labels):
        '''
        x : feature vector : (b x  d) b= batch size d = dimension 
        labels : (b,)
        '''
        # x = torch.rand(32,2048)
        # label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,])
        x = nn.functional.normalize(x, p=2, dim=1)  # normalize the features

        with torch.no_grad():
            self.weight.weight.div_(torch.norm(self.weight.weight, dim=1, keepdim=True))

        b = x.size(0)
        n = self.num_classes

        cos_angle = self.weight(x)
        cos_angle = torch.clamp(cos_angle, min=-1, max=1)
        for i in range(b):
            cos_angle[i][labels[i]] = cos_angle[i][labels[i]] - self.m
        weighted_cos_angle = self.s * cos_angle
        log_probs = self.CrossEntropy(weighted_cos_angle, labels)

        return log_probs
