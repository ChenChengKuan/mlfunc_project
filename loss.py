import torch.nn as nn
import torch.nn.functional as F
import math
import torch
class ArcFaceLoss(nn.modules.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.03):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):

        logits =  F.linear(F.normalize(inputs), F.normalize(self.weight))
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels_onehot = F.one_hot(labels, self.out_features)
        output = (labels_onehot * phi) + ((1.0 - labels_onehot) * cosine)
        output *= self.s

        return output

