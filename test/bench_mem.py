"""Benchmarks examining memory usage"""
import gc

from memory_profiler import memory_usage

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from interpn import MultilinearRectilinear, MultilinearRegular, MulticubicRegular
