"""Verify all archived files with only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    checksums = json.loads((root / 'SHA256SUMS.json').read_text())
    for name, expected in checksums.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or invalid package file: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('File checksum mismatch: ' + name)
    print('OK: ' + str(len(checksums)) + ' archived files verified')
    print('generated/ configs and rebuilt native bundles are not archived model replacements')


if __name__ == '__main__':
    main()
