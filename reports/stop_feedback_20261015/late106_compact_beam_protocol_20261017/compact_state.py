"""Offline branching with one immutable model and one reusable MuJoCo data.

Each frontier owns all Python history, RNG state, and the native integration
state. No arena/model copy is stored in a frontier. Exact physical/sensor
continuation is a mandatory admission check, not an assumed equivalence.
"""
from collections import deque
import copy
import ctypes
import hashlib
import random
import weakref

import mujoco
import numpy as np
import torch


OWN = ('experiments.', 'neural_navigation.', 'navrl.', '__main__', 'compact_state')


def model_digest(model):
    result = hashlib.sha256()
    for name in sorted(dir(model)):
        if name.startswith('_'):
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            result.update(str((name, value.dtype.str, value.shape)).encode())
            if value.size:
                result.update(memoryview(np.ascontiguousarray(value)).cast('B'))
        elif isinstance(value, (str, bytes, int, float, bool)):
            result.update(repr((name, value)).encode())
    result.update(repr(model.opt).encode())
    return result.hexdigest()


def clone(roots, bindings):
    memo = dict(bindings)
    objects, proxies, seen = [], [], set()

    def visit(value):
        identity = id(value)
        if identity in memo or identity in seen:
            return
        seen.add(identity)
        if type(value) in (weakref.ProxyType, weakref.CallableProxyType):
            proxies.append(value)
        elif isinstance(value, (torch.jit.ScriptModule, ctypes.CDLL, ctypes._CFuncPtr)):
            memo[identity] = value
        elif isinstance(value, (mujoco.MjModel, mujoco.MjData)):
            raise ValueError('Unbound mutable native object in compact snapshot')
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list, deque)):
            for item in value:
                visit(item)
        elif type(value).__module__.startswith(OWN) and hasattr(value, '__dict__'):
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
            raise ValueError('External weak reference in compact snapshot')
    for original in objects:
        memo[id(original)].__dict__.update(copy.deepcopy(vars(original), memo))
    return tuple(copy.deepcopy(root, memo) for root in roots)


class CompactSnapshot:
    def __init__(self, env, runtime):
        assert env.renderer is None
        self.model = env.model
        self.token = object()
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(self.model, self.spec))
        mujoco.mj_getState(self.model, env.data, self.state, self.spec)
        self.roots = clone((env, runtime), {id(self.model): self.model, id(env.data): self.token})
        self.numpy_rng = copy.deepcopy(np.random.get_state())
        self.python_rng = random.getstate()
        self.torch_rng = torch.get_rng_state().clone()

    def restore(self, reusable_data):
        mujoco.mj_setState(self.model, reusable_data, self.state, self.spec)
        mujoco.mj_forward(self.model, reusable_data)
        # Forward recomputes derived fields; restore integration/warmstart
        # values afterward so the next mj_step sees the captured initial state.
        mujoco.mj_setState(self.model, reusable_data, self.state, self.spec)
        roots = clone(self.roots, {id(self.model): self.model, id(self.token): reusable_data})
        np.random.set_state(copy.deepcopy(self.numpy_rng))
        random.setstate(self.python_rng)
        torch.set_rng_state(self.torch_rng.clone())
        return roots
