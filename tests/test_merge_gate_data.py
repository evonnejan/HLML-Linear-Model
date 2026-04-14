"""Tests for merge_gate_data.py — uses synthetic DataFrames, no real CSV files."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from merge_gate_data import GATE_COLS, STALENESS_LIMIT, merge_gate
