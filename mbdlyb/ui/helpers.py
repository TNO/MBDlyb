# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
from typing import Optional

import pandas as pd
from neomodel import DoesNotExist
from nicegui import ui
from itertools import product
from math import prod

from quimb.tensor.tensor_2d_tebd import conditioner

from mbdlyb.gdb import MBDElement, MBDNode


def goto(url: str):
	return ui.navigate.to(url)


# HELPER CLASSES

class Relation:
	_attr: str = ''

	_start_cls: type[MBDElement] = None
	_end_cls: type[MBDElement] = None
	start_uid: str = None
	end_uid: str = None
	_original_start_uid: Optional[str] = None
	_original_end_uid: Optional[str] = None
	weight: float

	def __init__(self, start_node: MBDElement = None, end_node: MBDElement = None, node_type: type[MBDElement] = None,
				 weight: float = 1.):
		if start_node is not None:
			self._start_cls = type(start_node)
			self.start_uid = start_node.uid
			self._original_start_uid = start_node.uid
		else:
			self._start_cls = node_type
		if end_node is not None:
			self._end_cls = type(end_node)
			self.end_uid = end_node.uid
			self._original_end_uid = end_node.uid
		else:
			self._end_cls = node_type
		self.weight = weight

	def get_start_node(self) -> MBDElement:
		return self._start_cls.nodes.get(uid=self.start_uid) if self.start_uid else None

	def get_end_node(self) -> MBDElement:
		return self._end_cls.nodes.get(uid=self.end_uid) if self.end_uid else None

	def get_original_start_node(self) -> MBDElement:
		return self._start_cls.nodes.get(uid=self._original_start_uid) if self._original_start_uid else None

	def get_original_end_node(self) -> MBDElement:
		return self._end_cls.nodes.get(uid=self._original_end_uid) if self._original_end_uid else None

	def start_node_changed(self) -> bool:
		return self._original_start_uid is not None and self.start_uid != self._original_start_uid

	def end_node_changed(self) -> bool:
		return self._original_end_uid is not None and self.end_uid != self._original_end_uid

	def save(self, redirect: str = None):
		if self.start_node_changed():
			self.get_original_start_node().__getattribute__(self._attr).disconnect(self.get_original_end_node())
		elif self._original_end_uid is not None:
			self.get_start_node().__getattribute__(self._attr).disconnect(self.get_original_end_node())
		self.get_start_node().__getattribute__(self._attr).connect(self.get_end_node(), self.get_props())
		if redirect is not None:
			ui.navigate.to(redirect)

	def get_props(self) -> dict:
		return {
			'weight': self.weight
		}


class CPTCell:
	line: 'CPTLine'
	probability: float

	def __init__(self, line: 'CPTLine', probability: float):
		self.line = line
		self.probability = probability

	@property
	def cpt(self) -> 'CPT':
		return self.line.cpt


class CPTLine:
	cpt: 'CPT'
	conditions: list[str]
	cells: list[CPTCell]

	def __init__(self, cpt: 'CPT', conditions: list[str], probabilities: list[float] = None):
		self.cpt = cpt
		self.conditions = conditions
		self.cells = [CPTCell(self, f) for f in (probabilities or ([0.] * len(self.values)))]

	def __repr__(self):
		return f'{", ".join(self.conditions)} - {" ".join(str(c.probability) for c in self.cells)}'

	@property
	def values(self) -> list[str]:
		return self.cpt.values

	def normalize(self) -> 'CPTLine':
		flatten = False
		total = sum(c.probability for c in self.cells)
		if total == 0.:
			flatten = True
			total = len(self.values)
		for c in self.cells:
			c.probability = (1. if flatten else c.probability) / total
		return self


class CPT:
	parents: list[MBDNode]
	node: MBDNode
	lines: list[CPTLine]

	def __init__(self, node: MBDNode, lines: list[CPTLine] = None):
		self.node = node
		self.lines = self.create_lines(lines)

	def __repr__(self):
		return 'CPT for {}\n{} - {}\n{}'.format(self.node, ", ".join(p.name for p in self.parents),
												", ".join(self.values), "\n".join(str(l) for l in self.lines))

	@property
	def parents(self) -> list[MBDNode]:
		return self.node.parents()

	@property
	def values(self) -> list[str]:
		return self.node.states

	def create_lines(self, _lines: list[list[float]] = None) -> list[CPTLine]:
		lines: list[CPTLine] = []
		condition_list = list(list(condition) for condition in product(*[p.states for p in self.parents]))
		if _lines is None:
			_lines = [[0.] * len(self.values) for _ in condition_list]
		for idx, conditions in enumerate(condition_list):
			lines.append(CPTLine(self, conditions, _lines[idx]))
		return lines

	def normalize(self) -> 'CPT':
		for line in self.lines:
			line.normalize()
		return self

	def set_prob(self, state: str, prob: float):
		state_idx = self.values.index(state)
		for line in self.lines:
			line.cells[state_idx].probability = prob

	def to_dict(self) -> dict:
		return {
			'state_count': len(self.values),
			'parents': [[p.fqn, len(p.states)] for p in self.parents],
			'lines': [[c.probability for c in l.cells] for l in self.lines]
		}

	def to_df(self) -> pd.DataFrame:
		parent_fqns = [p.fqn for p in self.parents]
		lines = [l.conditions + [c.probability for c in l.cells] for l in self.lines]
		return pd.DataFrame(data=lines, columns=parent_fqns + self.node.states).set_index(parent_fqns)

	@staticmethod
	def from_dict(node: MBDNode, d: dict) -> tuple['CPT', Optional[str]]:
		message = None
		if not d:
			message = 'No CPT found.'
		elif not all(p in d for p in ('state_count', 'parents', 'lines')):
			message = 'Invalid CPT provided.'
		elif d['state_count'] != len(node.states):
			message = 'Number of states has changed.'
		elif d['parents'] != [[p.fqn, len(p.states)] for p in node.parents()]:
			message = 'Dependencies have changed.'
		elif len(d['lines']) != prod(len(p.states) for p in node.parents()):
			message = 'Number of stored CPT lines is invalid.'
		if message is not None:
			return CPT(node), message
		return CPT(node, d['lines']), None

	@staticmethod
	def from_df(node: MBDNode, df: pd.DataFrame) -> tuple['CPT', Optional[str]]:
		message = None
		if list(df.columns) != node.states:
			message = f'CPT-provided states do not match expected states ({", ".join(node.states)}).'
		elif list(df.index.names) != [p.fqn for p in node.parents()]:
			message = f'CPT-provided parents do not match the expected parents.'
		elif len(df.index) != prod(len(p.states) for p in node.parents()):
			message = 'Number of provided CPT lines is invalid. This is likely caused by a change in the states of one or more of its parents.'
		elif df.sum().sum() == 0.:
			message = 'Unfilled CPT was provided.'
		if message is not None:
			return CPT(node), message
		return CPT(node, df.values.tolist()), None


# DATABASE HELPERS

def get_object_or_404[T](cls: type[T], **kwargs) -> T:
	try:
		return cls.nodes.get(**kwargs)
	except DoesNotExist:
		ui.label(f'Could not find {cls.__name__} with parameters {dict(**kwargs)}.')
		return None


def get_relation_or_404(start_node: MBDElement, end_node: MBDElement, attr: str):
	relation = start_node.__getattribute__(attr).relationship(end_node)
	if relation is None:
		ui.label(f'Could not find \'{attr}\' relationship between {start_node} and {end_node}.')
	return relation
