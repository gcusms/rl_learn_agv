"""Integrity checks before loading an opt-in localization implementation."""
import hashlib
import json
import platform
from pathlib import Path


def check_localization_files(bundle, manifest):
    bundle = Path(bundle)
    for optional in ('localization_contract.json', 'localization_kernel.so', 'uncertainty_contract.json'):
        if (bundle / optional).exists() and optional not in manifest['files']:
            raise ValueError('Missing localization component hash: ' + optional)
    for name, expected in manifest['files'].items():
        if Path(name).name != name or hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Bundle integrity failure: ' + name)


def localization_configuration(bundle):
    """Require a pinned bundle for native code; leave legacy bundles unchanged.

    Python-only NavigationCore fixtures may supply just the explicit contract
    and router. A native library is always checked against a manifest before
    ctypes can load it. This branch opts into geometric tracking only.
    """
    bundle = Path(bundle)
    path = bundle / 'localization_contract.json'
    if not path.exists():
        if any((bundle / name).exists() for name in ('localization_kernel.so', 'uncertainty_contract.json')):
            raise ValueError('Localization component without a localization contract')
        return None
    config = json.loads(path.read_text())
    from neural_navigation.fast_static_localizer import (
        contract, native_contract, sparse_contract, temporal_contract,
    )
    supported = (contract(), native_contract(), sparse_contract(), sparse_contract(True),
                 temporal_contract(), temporal_contract(True))
    if config not in supported:
        raise ValueError('Unsupported localization contract in this release')
    manifest_path = bundle / 'manifest.json'
    native = 'kernel' in config
    if native or manifest_path.exists():
        if not manifest_path.exists():
            raise ValueError('Native localization requires a manifest')
        manifest = json.loads(manifest_path.read_text())
        check_localization_files(bundle, manifest)
        if native:
            if 'localization_kernel.so' not in manifest['files']:
                raise ValueError('Native localization requires a pinned kernel')
            architecture = manifest.get('native_kernel', {}).get('architecture')
            if architecture != platform.machine():
                raise ValueError('Localization kernel architecture mismatch; rebuild on the target platform')
    return config
