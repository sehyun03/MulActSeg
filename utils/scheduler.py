from torch.optim.lr_scheduler import _LRScheduler, StepLR
import math

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max(base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

def ramp_up(x, lamparam=0.1, scale=1.0, dorampup=True):
    r""" Adaptive loss weight scheduler
         lamparam(float): weight increase damping ratio
         scale(float): final weight
         """
    if not dorampup or x > 1.0:
        return 1.0

    return sigmoid_ramp_up(x, lamparam, scale)

def sigmoid_ramp_up(x, lamparam, scale):
    den = 1.0 + math.exp(-x/lamparam) # for low increase ratio
    lamb = 2.0 / den - 1.0
    return lamb * scale