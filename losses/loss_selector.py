from a_softmax import AngularPenaltySMLoss
from am_softmax import AM_Softmax
from center_loss import CenterLoss
from circle import CircleLoss
from contrastive_loss import ContrastiveLoss
from contrastive_center_loss import ContrastiveCenterLoss
from focal_loss import FocalLoss
from triplet import TripletLoss
from triplet_center import TripletCenterLoss


class Loss_Selector(object):
    def __init__(self):
        self.loss = None
        self.kwargs = None

    def select_loss(self, name, **kwargs):
        method = getattr(self, name, lambda: "Invalid loss")
        self.kwargs = kwargs
        return method()

    def a_softmax(self):
        self.loss = AngularPenaltySMLoss(self.kwargs)
        print("Choosing AngularPenaltySMLoss...")
        return self.loss

    def am_softmax(self):
        self.loss = AM_Softmax(self.kwargs)
        print("Choosing AM_softmax...")
        return self.loss

    def center(self):
        self.loss = CenterLoss(self.kwargs)
        print("Choosing Center Loss...")
        return self.loss

    def circle(self):
        self.loss = CircleLoss(self.kwargs)
        print("Choosing Circle Loss...")
        return self.loss

    def contrastive(self):
        self.loss = ContrastiveLoss(self.kwargs)
        print("Choosing Contrastive Loss...")
        return self.loss

    def contrast_center(self):
        self.loss = ContrastiveCenterLoss(self.kwargs)
        print("Choosing Contrastive Center Loss...")
        return self.loss

    def focal(self):
        self.loss = FocalLoss(self.kwargs)
        print("Choosing Focal loss...")
        return self.loss

    def triplet(self):
        self.loss = TripletLoss(self.kwargs)
        print("Choosing Triplet Loss...")
        return self.loss

    def triplet_center(self):
        self.loss = TripletCenterLoss(self.kwargs)
        print("Choosing Triplet Center Loss...")
        return self.loss


