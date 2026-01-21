# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
from mbdlyb.gdb import MBDElement
from mbdlyb.functional.gdb import Cluster
from mbdlyb.ui.validation import valid_name


def base_name_validation(cluster: Cluster = None, element: MBDElement = None, allow_empty: bool = False):
	validation = {
		'Invalid characters used': valid_name
	}
	if not allow_empty:
		validation['Required field'] = lambda s: s != ''
	elements = set(cluster.elements.all()) if cluster is not None else {c for c in Cluster.nodes.has(net=False)}
	children = {e.name for e in elements if element is None or e != element}
	validation['Name already exists'] = lambda s: s not in children
	return validation
