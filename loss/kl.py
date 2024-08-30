import torch.nn.functional as F
import torch

class KLLoss:
    """ KL Divergence"""

    def __init__(self, t=4, bidir=False):
        self.t = t
        self.bidir = bidir

    def uni_direct(self, logits_s, logits_t):
        p_s = F.log_softmax(logits_s / self.t, dim=1)
        p_t = F.softmax(logits_t / self.t, dim=1)
        if logits_s.shape[0] != 0:
            loss = F.kl_div(p_s, p_t, reduction='sum') * (self.t ** 2) / logits_s.shape[0]
        else:
            loss = torch.tensor([0.0]).to(logits_s.device)

        # print('kl loss: ', loss)
        return loss

    def bi_direct(self, logits_s, logits_t):
        p_s = F.log_softmax(logits_s / self.t, dim=1)
        p_t = F.log_softmax(logits_t / self.t, dim=1)
        if logits_s.shape[0] != 0:
            loss = F.kl_div(p_s, p_t, reduction='sum', log_target=True) * (self.t ** 2) / logits_s.shape[0] + \
                   F.kl_div(p_t, p_s, reduction='sum', log_target=True) * (self.t ** 2) / logits_s.shape[0]
        else:
            loss = torch.tensor([0.0]).to(logits_s.device)

        # print('kl loss: ', loss)
        return loss

    def __call__(self, logits_s, logits_t):
        if self.bidir:
            loss = self.bi_direct(logits_s, logits_t)
        else:
            loss = self.uni_direct(logits_s, logits_t)
        return loss