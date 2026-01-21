# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
import re


def valid_name(s: str) -> bool:
	return bool(re.match('^[a-zA-Z0-9_-]*$', s))
