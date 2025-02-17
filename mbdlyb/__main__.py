# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from .functional.ui import run as run_functional

if __name__ in {"__main__", "__mp_main__"}:
	run_functional()
