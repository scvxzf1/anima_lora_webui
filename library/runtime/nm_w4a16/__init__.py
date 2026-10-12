"""Opt-in 2:4 INT4/BF16 experiment; not a production training backend.

Import individual tensor modules explicitly. The cost model can be imported
without importing torch or initializing CUDA.
"""
