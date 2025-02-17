# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import re


def valid_name(s: str) -> bool:
	return bool(re.match('^[a-zA-Z0-9_-]*$', s))
