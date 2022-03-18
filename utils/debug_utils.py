import torch
import time
from collections import OrderedDict


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=True):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.start = time.time()
        self.last_time = time.time()
        self.cuda = False
        self.add = True
        self.reset()

    def reset(self):
        if self.cuda:
            torch.cuda.synchronize()
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        if not self.add:
            return
        if self.cuda:
            torch.cuda.synchronize()
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()