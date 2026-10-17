"""Rebuild an archived native localization bundle into a new directory."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(source, output, eigen_include='/usr/include/eigen3'):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    if 'localization_kernel.so' not in manifest['files'] or 'native_kernel' not in manifest:
        raise ValueError('Expected an archived native bundle')
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid component name')
        if name != 'localization_kernel.so' and digest(source / name) != expected:
            raise ValueError('Source component mismatch: ' + name)
    kernel = ROOT / 'training/neural_navigation/static_tracking_kernel.cpp'
    if digest(kernel) != manifest['native_kernel']['source_sha256']:
        raise ValueError('Native source differs from archived bundle')
    compiler = shutil.which('g++')
    if not compiler or not (Path(eigen_include) / 'Eigen/Dense').is_file():
        raise RuntimeError('Install g++ and libeigen3-dev, or supply --eigen-include')
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('*.so', 'MODEL_CARD.json'))
    command = [compiler, '-O3', '-DNDEBUG', '-std=c++17', '-shared', '-fPIC',
               str(kernel), '-I' + str(eigen_include), '-o', str(output / 'localization_kernel.so')]
    subprocess.run(command, check=True)
    manifest['files']['localization_kernel.so'] = digest(output / 'localization_kernel.so')
    manifest['native_kernel'].update(architecture=platform.machine(),
                                    source_sha256=digest(kernel), runtime_compilation=False)
    manifest['deployment_approved'] = False
    manifest['handoff_rebuild'] = dict(parent_manifest_sha256=digest(source / 'manifest.json'),
                                     platform=platform.platform(), compiler=subprocess.check_output(
                                         [compiler, '-dumpfullversion'], text=True).strip())
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eigen-include', default='/usr/include/eigen3')
    args = parser.parse_args()
    print(build(args.source, args.output, args.eigen_include))
