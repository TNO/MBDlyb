# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
from collections import OrderedDict
from itertools import product
from typing import Callable, Type
from math import ceil
from more_itertools import divide

import pandas as pd
import quimb.tensor as qtn
import numpy as np
from scipy.stats import entropy

from mbdlyb import MBDElement, MBDNode, MBDNet, MBDRelation, MBDReasoner


MAX_PARENTS = 10


class TNRelation(MBDRelation):
	pass


class TNElement(MBDElement):
	def create_tn_nodes(self, tn: qtn.TensorNetwork):
		raise NotImplementedError(f'Function \'create_tn_node\' is not implemented for {self.__class__.name}!')

	def create_tn_connections(self, tn: qtn.TensorNetwork):
		raise NotImplementedError(f'Function \'create_tn_connections\' is not implemented for {self.__class__.name}!')


class TensorNode(TNElement, MBDNode):
	_array_fn: Callable[[...], list[float]] = None

	cpt_parents: list['TensorNode'] = None
	cpt: list[list[float]] = None

	@property
	def has_cpt(self) -> bool:
		return self.cpt is not None

	@property
	def parents_tn(self) -> list['TensorNode']:
		return self.cpt_parents or [pr.source for pr in self.parent_relations if isinstance(pr, TNRelation)]

	def create_tn_nodes(self, tn: qtn.TensorNetwork):
		"""Creates a tensor node inside the tensor network tn"""

		# add this node to the TN
		edges = (self.parents_tn if self.has_cpt else list(reversed([p.fqn for p in self.parents_tn]))) + [self.fqn]
		tn.add_tensor(qtn.Tensor(data=self._compute_array(), inds=edges), tid=self.fqn + 'tensor')

	def create_tn_connections(self, tn: qtn.TensorNetwork):
		"""Adds edges between this node and the rest of the nodes in the network. I'm not sure this method is necessary"""
		pass

	def set_array_fn(self, array_fn: Callable[..., list[float]] | None):
		self._array_fn = array_fn

	def _compute_array(self) -> np.ndarray:
		if self.has_cpt:
			shape = [len(p.states) for p in self.parents_tn] + [len(self.states)]
			table = np.array(self.cpt)
		else:
			shape = [len(p.states) for p in reversed(self.parents_tn)] + [len(self.states)]
			params = list(product(*[[(p, v) for v in p.states] for p in reversed(self.parents_tn)]))
			table = np.array([self._compute_array_line(dict(ps)) for ps in params]).flatten()
		return np.reshape(table, shape)

	def _compute_array_line(self, values: dict[MBDNode, str]) -> list[float]:
		if self._array_fn is not None:
			return self._array_fn(values)
		raise NotImplementedError(f'Method \'_compute_array_line\' is not implemented!')


class TensorAutoSplitNode(TensorNode):
	_split_preserve_classes: list[Type[TensorNode]] = []
	_parent_split: OrderedDict[str, list[TensorNode]] = None

	def create_tn_nodes(self, tn: qtn.TensorNetwork):
		if len(self.parents_tn) > MAX_PARENTS:
			self._parent_split = OrderedDict()
			fixed_parents: list[TensorNode] = []
			eligable_parents_for_split: list[TensorNode] = []
			for p in self.parents_tn:
				if isinstance(p, *self._split_preserve_classes):
					fixed_parents.append(p)
				else:
					eligable_parents_for_split.append(p)

			split_room = max(2, MAX_PARENTS - len(fixed_parents))
			count_groups = ceil(len(eligable_parents_for_split) / split_room)
			for idx, g in enumerate(divide(count_groups, eligable_parents_for_split)):
				self._parent_split[f'{self.fqn}__{idx + 1}'] = fixed_parents + list(g)

			# add group nodes to the TN
			for group_id, parents in self._parent_split.items():
				edges = list(reversed([p.fqn for p in parents])) + [group_id]
				tn.add_tensor(qtn.Tensor(data=self._compute_limited_array(parents), inds=edges), tid=group_id + 'tensor')
			# add main node to the TN
			edges = list(reversed([p for p in self._parent_split.keys()])) + [self.fqn]
			tn.add_tensor(qtn.Tensor(data=self._compute_join_array(len(self._parent_split)), inds=edges), tid=self.fqn + 'tensor')

		else:
			super().create_tn_nodes(tn)

	def _compute_limited_array(self, parents: list[TensorNode]) -> np.ndarray:
		shape = [len(p.states) for p in reversed(parents)] + [len(self.states)]
		params = list(product(*[[(p, v) for v in p.states] for p in reversed(parents)]))
		factor = np.array([self._compute_array_line(dict(ps)) for ps in params]).flatten()
		return np.reshape(factor, shape)

	def _compute_join_array(self, split_count: int) -> np.ndarray:
		raise NotImplementedError


class TensorNet(TNElement, MBDNet):
	nodes: dict[str, TNElement] = None
	_edges: dict[str] = None

	def get_tn_nodes(self) -> list[TNElement]:
		return [n for n in self.nodes.values() if isinstance(n, TNElement)]

	def to_tn(self) -> qtn.TensorNetwork:
		"""Takes an abstract MBDNet (a knowledge graph) and turns it into a quimb tensor network"""
		tn = qtn.TensorNetwork([])
		with self.create_reasoner_view():
			self.create_tn_nodes(tn)
		return tn

	def create_tn_nodes(self, tn: qtn.TensorNetwork):
		# This recursive relation might not be needed for quimb because it natively allows to add a TN to a TN.
		for node in self.get_tn_nodes():
			node.create_tn_nodes(tn)

	def create_tn_connections(self, tn: qtn.TensorNetwork):
		for node in self.get_tn_nodes():
			node.create_tn_connections(tn)


class TensorNetReasoner(MBDReasoner):
	_tn: qtn.TensorNetwork = None

	def __init__(self, net: TensorNet):
		super().__init__(net)
		self.init()

	def init(self):
		self._tn = self._net.to_tn()

	def add_evidence(self, evidence: dict[str, str | dict[str, float]], reset: bool=False):
		if reset:
			self.drop_evidence()
		super().add_evidence(evidence)
		for n, ev in self.evidence.items():
			if n in self._tn.tensor_map.keys():
				self._tn.pop_tensor(n)
			self._tn.add_tensor(qtn.Tensor(data=np.array(ev), inds=[n], tags=('Evidence', n)), tid=n)

	def drop_evidence(self, *evs: str):
		if len(evs) > 0:
			for e in evs:
				self._tn.delete(tags=((e, "Evidence")), which='all')
		else:
			if "Evidence" in self._tn.tags:
				self._tn.delete(tags=("Evidence"), which='all')
		super().drop_evidence(*evs)

	def infer(self, *targets: str) -> pd.DataFrame:
		targets = [t for t in (set(targets) or self._targets)]
		simplified_tn = self._tn
		posteriors = pd.DataFrame()
		for target in targets:
			target_net = simplified_tn.contract(output_inds=[target])  
			target_vals = (target_net.data / sum(target_net.data))
			states = self._net.get_node(target).states
			target_dict = dict(zip(states, target_vals))
			target_df = pd.DataFrame(data=target_dict, index=[target])
			posteriors = pd.concat([posteriors, target_df])
		return posteriors

	def is_d_separated(self, Y: str | set[str], X: str | set[str], Z: str | set[str] | dict[str, str] = None):
		raise NotImplementedError('Method \'is_d_separated\' is not implemented for TensorNetwork!')

	def conditional_entropy(self, Y: set[str], X: set[str], evidence: dict[str, str] = None) -> float:
		Xs, Ys = list(X), list(Y)
		XY = Xs + Ys

		# TODO: Need to properly restore state after adding evidence

		XY_posterior_tn = self._tn.contract(output_inds=XY)
		X_posterior_tn = XY_posterior_tn.contract(output_inds=Xs)

		h_xy = entropy(XY_posterior_tn.data.flatten(), base=2)
		h_x = entropy(X_posterior_tn.data.flatten(), base=2)
		return h_xy - h_x

	def entropy(self, Y: set[str], evidence: dict[str, str] = None) -> float:
		Ys = list(Y)

		# TODO: Need to properly restore state after adding evidence

		posterior_tn = self._tn.contract(output_inds=Ys)

		return entropy(posterior_tn.data.flatten(), base=2)