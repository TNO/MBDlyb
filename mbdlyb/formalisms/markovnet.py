# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from collections import OrderedDict
from itertools import product
from more_itertools import divide
from math import ceil
from typing import Callable, Type
import pyagrum as gum
import numpy as np
import pandas as pd

from mbdlyb import MBDElement, MBDNode, MBDNet, MBDRelation, MBDReasoner


MAX_PARENTS = 10


class MNRelation(MBDRelation):
	pass


class MNElement(MBDElement):
	def create_mn_node(self, mn: gum.MarkovRandomField, node_sets: dict[Type[MBDNode], list[int]]):
		raise NotImplementedError(f'Function \'create_mn_node\' is not implemented for {self.__class__.name}!')

	def create_mn_connections(self, mn: gum.MarkovRandomField):
		raise NotImplementedError(f'Function \'create_mn_connections\' is not implemented for {self.__class__.name}!')


class MarkovNode(MNElement, MBDNode):
	_factor_fn: Callable[[dict['MBDNode', str]], list[float]] = None

	cpt_parents: list['MarkovNode'] = None
	cpt: list[list[float]] = None

	@property
	def has_cpt(self) -> bool:
		return self.cpt is not None

	@property
	def parents_mn(self) -> list['MarkovNode']:
		return self.cpt_parents or [pr.source for pr in self.parent_relations if isinstance(pr, MNRelation)]

	def create_mn_node(self, mn: gum.MarkovRandomField, node_sets: dict[Type[MBDNode], list[int]]):
		if mn is None:
			raise AssertionError('A MarkovNet must be provided to add a node to it!')

		# add this node to the MN
		id: int = mn.add(gum.LabelizedVariable(self.fqn, self.name, self.states))
		if self.__class__ in node_sets:
			node_sets[self.__class__].append(id)
		else:
			node_sets[self.__class__] = [id]

	def create_mn_connections(self, mn: gum.MarkovRandomField):
		nodes = [self.fqn] + [p.fqn for p in (reversed(self.parents_mn) if self.has_cpt else self.parents_mn)]
		mn.addFactor(nodes)
		mn.factor(nodes)[:] = self._compute_factor()

	def set_factor_fn(self, factor_fn: Callable[..., list[float]] | None):
		self._factor_fn = factor_fn

	def _compute_factor(self) -> np.ndarray:
		if self.has_cpt:
			shape = [len(p.states) for p in self.parents_mn] + [len(self.states)]
			factor = np.array(self.cpt)
		else:
			shape = [len(p.states) for p in reversed(self.parents_mn)] + [len(self.states)]
			params = list(product(*[[(p, v) for v in p.states] for p in reversed(self.parents_mn)]))
			factor = np.array([self._compute_factor_line(dict(ps)) for ps in params]).flatten()
		return np.reshape(factor, shape)

	def _compute_factor_line(self, values: dict[MBDNode, str]) -> list[float]:
		if self._factor_fn is not None:
			return self._factor_fn(values)
		raise NotImplementedError(f'Method \'_compute_factor_line\' is not implemented!')


class MarkovAutoSplitNode(MarkovNode):
	_split_preserve_classes: list[Type[MarkovNode]] = []
	_parent_split: OrderedDict[str, list[MarkovNode]] = None

	def create_mn_node(self, mn: gum.MarkovRandomField, node_sets: dict[Type[MBDNode], list[int]]):
		super().create_mn_node(mn, node_sets)

		if len(self.parents_mn) > MAX_PARENTS:
			self._parent_split = OrderedDict()
			fixed_parents: list[MarkovNode] = []
			eligable_parents_for_split: list[MarkovNode] = []
			for p in self.parents_mn:
				if isinstance(p, *self._split_preserve_classes):
					fixed_parents.append(p)
				else:
					eligable_parents_for_split.append(p)

			split_room = max(2, MAX_PARENTS - len(fixed_parents))
			count_groups = ceil(len(eligable_parents_for_split) / split_room)
			for idx, g in enumerate(divide(count_groups, eligable_parents_for_split)):
				self._parent_split[f'{self.fqn}__{idx + 1}'] = fixed_parents + list(g)

			# add group nodes to the MN
			for group_id in self._parent_split.keys():
				id: int = mn.add(gum.LabelizedVariable(group_id, self.name + f'__{group_id}', self.states))
				node_sets[self.__class__].append(id)

	def create_mn_connections(self, mn: gum.MarkovRandomField):
		if self._parent_split is None:
			super().create_mn_connections(mn)
		else:
			split_nodes = [self.fqn] + list(self._parent_split.keys())
			mn.addFactor(split_nodes)
			mn.factor(split_nodes)[:] = self._compute_join_factor(len(self._parent_split))

			for split_node, parents in self._parent_split.items():
				nodes = [split_node] + [p.fqn for p in parents]
				mn.addFactor(nodes)
				mn.factor(nodes)[:] = self._compute_limited_factor(parents)

	def _compute_limited_factor(self, parents: list[MarkovNode]) -> np.ndarray:
		shape = [len(p.states) for p in reversed(parents)] + [len(self.states)]
		params = list(product(*[[(p, v) for v in p.states] for p in reversed(parents)]))
		factor = np.array([self._compute_factor_line(dict(ps)) for ps in params]).flatten()
		return np.reshape(factor, shape)

	def _compute_join_factor(self, split_count: int) -> np.ndarray:
		raise NotImplementedError


class MarkovNet(MNElement, MBDNet):
	def get_mn_nodes(self) -> list[MNElement]:
		return [n for n in self.nodes.values() if isinstance(n, MNElement)]

	def to_mn(self) -> tuple[gum.MarkovRandomField, dict[Type[MBDNode], list[int]]]:
		mn = gum.MarkovRandomField(self.name)
		node_sets: dict[Type[MBDNode], list[int]] = dict()

		with self.create_reasoner_view():
			# add all nodes to the MRF
			self.create_mn_node(mn, node_sets)
			# add connections to the MRF
			self.create_mn_connections(mn)
		return mn, node_sets

	def create_mn_node(self, mn: gum.MarkovRandomField, node_sets: dict[Type[MBDNode], list[int]]):
		for node in self.get_mn_nodes():
			node.create_mn_node(mn, node_sets)

	def create_mn_connections(self, mn: gum.MarkovRandomField):
		for node in self.get_mn_nodes():
			node.create_mn_connections(mn)


class MarkovNetReasoner(MBDReasoner):
	_mn: gum.MarkovRandomField = None
	_ie: gum.ShaferShenoyMRFInference = None

	def __init__(self, net: MarkovNet):
		super().__init__(net)
		self.init()

	def init(self):
		self._mn, _ = self._net.to_mn()
		self._ie = gum.ShaferShenoyMRFInference(self._mn)

	def infer(self, *targets: str) -> pd.DataFrame:
		self._ie.setEvidence(self._evidence)
		self._ie.makeInference()
		targets = set(targets) or self._targets
		return pd.concat([
			self._ie.posterior(target).topandas().unstack(level=1)
			for target in targets
		])

	def conditional_entropy(
			self, y: set[str], x: set[str], evidence: dict[str, str] = None
	) -> float:
		"""Computes entropy of y given x."""
		ie = gum.ShaferShenoyMRFInference(self._mn)

		evidence = evidence or self._evidence
		if evidence:
			ie.setEvidence(evidence)

		it = gum.InformationTheory(ie, x, y)
		return it.entropyYgivenX()

	def entropy(self, x: set[str], evidence: dict[str, str] = None) -> float:
		"""Computes entropy of a set of nodes."""
		ie = gum.ShaferShenoyMRFInference(self._mn)
		ie.addJointTarget(x)

		evidence = evidence or self._evidence
		if evidence:
			ie.setEvidence(evidence)

		ie.makeInference()
		return ie.jointPosterior(x).entropy()
