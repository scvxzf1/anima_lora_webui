"""Tiny valid ordinary checkpoints for Web preflight path tests."""
import struct


def write_ordinary_checkpoint(path):
    header = b'{}      '
    path.write_bytes(struct.pack('<Q', len(header)) + header)
