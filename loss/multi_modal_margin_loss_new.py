import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable


class multiModalMarginLossNew(nn.Module):
    def __init__(self, margin=3, dist_type='cos'):
        super(multiModalMarginLossNew, self).__init__()
        self.dist_type = dist_type
        self.margin = margin
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat0, feat1, feat2, feat3, label1):
        # print("using 3MLoss")
        # print(feat1.shape, feat2.shape, label1.shape, label1)
        label_num = len(label1.unique())
        feat0 = feat0.chunk(label_num, 0)
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        feat3 = feat3.chunk(label_num, 0)
        # loss = Variable(.cuda())
        # print(label_num)
        # print(feat1)
        # assert 1 < 0
        for i in range(label_num):
            center0 = torch.mean(feat0[i], dim=0)
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            center3 = torch.mean(feat3[i], dim=0)
            if i == 0:
              # dist = max(self.margin - self.dist(center0, center1, 0), abs(self.margin - self.dist(center0, center2)), abs(self.margin - self.dist(center0, center3)))
              dist = max(self.margin - self.dist(center0, center1), 0) + max(self.margin - self.dist(center0, center2), 0) + max(self.margin - self.dist(center0, center3), 0)
            else:
              # dist += max(abs(self.margin - self.dist(center0, center1)), abs(self.margin - self.dist(center0, center2)), abs(self.margin - self.dist(center0, center3)))
              dist += max(self.margin - self.dist(center0, center1), 0) + max(self.margin - self.dist(center0, center2), 0) + max(self.margin - self.dist(center0, center3), 0)

        return dist



