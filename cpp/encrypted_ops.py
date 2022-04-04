import torch
import functools
import torch

import numpy as np


import functools
import torch as phe

HANDLED_FUNCTIONS  ={}
def encryted_tensor():
    def decorator_func(func):
        def wrapper_func(*args, **kwargs):
            # Invoke the wrapped function first
            retval = func(*args, **kwargs)
            # Now do something here with retval and/or action
            assert not isinstance(args[0]._tensor , EncryptedTensor)
            return EncryptedTensor(retval, args[0].q)
        return wrapper_func
    return decorator_func

class EncryptedTensor(object):
    def __init__(self, tensor, q = 2**20):
        self.q = q
        self._tensor = tensor
    def __repr__(self):
        return "{}".format(self._tensor)
    def tensor(self):
        return self._tensor
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
          issubclass(t, (torch.Tensor, EncryptedTensor))
          for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)
    @encryted_tensor()
    def __add__(self, other):
        return phe.add(self._tensor, other._tensor)
    @encryted_tensor()
    def __sub__(self, other):
        return phe.sub(self._tensor, other._tensor)
    @encryted_tensor()
    def __mul__(self, other):
        return phe.mul(self._tensor,other._tensor)//self.q
    @encryted_tensor()
    def matmul(self, other):
        return phe.matmul(self._tensor,other._tensor)//self.q
    @encryted_tensor()
    def __div__(self, other):
        return phe.div(self._tensor, other._tensor)
    @encryted_tensor()
    def __mm__(self, other):
        return phe.mul(self._tensor, other._tensor)//self.q
    @encryted_tensor()
    def __lt__(self, other):
        return self._tensor <other
    @encryted_tensor()
    def __le__(self, other):
        return self._tensor <= other
    @encryted_tensor()
    def __gt__(self, other):
        return self._tensor >other
    @encryted_tensor()
    def __ge__(self, other):
        return self._tensor >=other
    @encryted_tensor()
    def squeeze(self, axis):
        return self._tensor.squeeze(axis)
    @encryted_tensor()
    def unsqueeze(self, axis):
        return self._tensor.unsqueeze(axis)        
    def __len__(self): return len(self._tensor)
    @encryted_tensor()
    def clone(self): return self._tensor.clone()
    @encryted_tensor()
    def long(self): return self._tensor.long()
    def size(self, axis): return self._tensor.size(axis)
    @encryted_tensor()
    def repeat(self, *args): return self._tensor.repeat(*args)
    @encryted_tensor()
    def reshape(self, *args): return self._tensor.reshape(*args)
    @encryted_tensor()
    def permute(self, *args): return self._tensor.permute(*args)
    def argmax(self, *args, **kwargs): return self._tensor.argmax(*args,  **kwargs)
    @encryted_tensor()
    def max(self, other): return phe.max(self._tensor, other._tensor)
    @encryted_tensor()
    def sum(self, *args): return self._tensor.sum(*args)
    @encryted_tensor()
    def flatten(self): return self._tensor.flatten()
    @encryted_tensor()
    def encrypt(self): return phe.encrypt(self._tensor)
    @encryted_tensor()
    def decrypt(self): return phe.decrypt(self._tensor)
    @encryted_tensor()
    def __getitem__(self, i): return self._tensor[i]
    def __setitem__(self, i, val):
        if isinstance(i, EncryptedTensor):
            self._tensor[i._tensor] = val
        else:
            self._tensor[i] = val
    @encryted_tensor()
    def t(self): return self._tensor.T

def implements(torch_function):
    """Register a torch function override for ScalarTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator
@implements(torch.div)
def div(input, other, rounding_mode = 'floor'):
    return EncryptedTensor(phe.div(input._tensor ,  other, rounding_mode = rounding_mode), input.q)
