"""In-process branches of headless recovery physics and sensor-runtime state.

Mutable history, MuJoCo data/model, maps, caches and RNGs are copied. Only
stateless native entry points and checked, unchanged TorchScript models share
storage. This accelerates repeated experiments; it changes no action or guard.
It is not a serialized deployment checkpoint or an independent success trial.
"""
from collections import deque
import copy
import ctypes
import hashlib
import random
import weakref

import numpy as np
import torch


OWN_PACKAGES = ('experiments.', 'neural_navigation.', 'navrl.')


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((name, str(value.dtype), value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def clone_roots(roots):
    """Preallocate forwarding wrappers before copying their linked dictionaries.

    Ordinary deepcopy can recursively query an uninitialized __getattr__
    wrapper, and ctypes native handles cannot be pickled. Preallocating the
    project objects also preserves core/guard/policy and weak-env references.
    """
    memo, objects, proxies, modules, seen = {}, [], [], [], set()

    def visit(value):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if type(value) in (weakref.ProxyType, weakref.CallableProxyType):
            proxies.append(value)
            return
        if isinstance(value, torch.jit.ScriptModule):
            memo[identity] = value
            modules.append(value)
            return
        if isinstance(value, (ctypes.CDLL, ctypes._CFuncPtr)):
            memo[identity] = value
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list, deque)):
            for item in value:
                visit(item)
        elif (type(value).__module__.startswith(OWN_PACKAGES)
                or type(value).__module__ == '__main__') and hasattr(value, '__dict__'):
            memo[identity] = object.__new__(type(value))
            objects.append(value)
            visit(vars(value))

    for root in roots:
        visit(root)
    for proxy in proxies:
        for original in objects:
            if proxy.__dict__ is vars(original):
                memo[id(proxy)] = weakref.proxy(memo[id(original)])
                break
        else:
            raise ValueError('Unmapped external weak reference in recovery snapshot')
    # Every project shell is now in memo, so cross-references resolve even
    # before that object's own dictionary is populated.
    for original in objects:
        memo[id(original)].__dict__.update(copy.deepcopy(vars(original), memo))
    return tuple(copy.deepcopy(root, memo) for root in roots), modules


class RecoverySnapshot:
    def __init__(self, env, runtime):
        if getattr(env, 'renderer', None) is not None:
            raise ValueError('Recovery snapshots require a headless environment')
        self.numpy_rng = copy.deepcopy(np.random.get_state())
        self.python_rng = random.getstate()
        self.torch_rng = torch.get_rng_state().clone()
        self.roots, shared = clone_roots((env, runtime))
        self.shared_modules = [(module, tensor_digest(module)) for module in shared]

    def verify_shared_models(self):
        for module, expected in self.shared_modules:
            if tensor_digest(module) != expected:
                raise ValueError('A shared inference model changed after snapshot creation')

    def restore(self):
        self.verify_shared_models()
        roots, _ = clone_roots(self.roots)
        np.random.set_state(copy.deepcopy(self.numpy_rng))
        random.setstate(self.python_rng)
        torch.set_rng_state(self.torch_rng.clone())
        return roots
