# Vendor
import numpy as np

# Project
from util import encode

class Task(object):
    """ Base class for all task of NRAM """

    def __init__(self, batch_size: int, max_int: int, num_regs: int, timestep: int) -> None:
        self.batch_size = batch_size
        self.max_int = max_int
        self.num_regs = num_regs
        self.timesteps = timestep

    def __call__(self, *args, **kwargs):
        in_mem, out_mem, error_mask = self.create()
        return encode(in_mem), \
               out_mem, \
               error_mask, \
               self.init_regs(np.zeros((self.batch_size, self.num_regs, self.max_int), dtype=np.float64)), \
               self.timesteps

    def __str__(self):
        return self.__class__.__name__

    def init_regs(self, regs):
        regs[:, :, 0] = 1.0
        return regs

    def create(self):
        raise NotImplementedError()