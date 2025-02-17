# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import json
import warnings
import re
from pathlib import Path
from typing import Optional, Type

import pandas as pd
from neomodel import (FloatProperty, ArrayProperty, StringProperty, ZeroOrOne, RelationshipTo, RelationshipFrom,
					  JSONProperty, db)

import mbdlyb.functional as fn
from mbdlyb import longest_common_fqn
from mbdlyb.operating_mode import OpmLogic, OpmSet, OpmVariable
from mbdlyb.gdb import MBDRelation, MBDNode, LBL_PARTOF, PartOfRelation, MBDNet, MBDElement


LBL_SUBFUNCTION = 'SUBFUNCTION_OF'
LBL_REQUIRED = 'REQUIRED_FOR'
LBL_REALIZES = 'REALIZES'
LBL_AFFECTS = 'AFFECTS'
LBL_OBSERVED = 'OBSERVED_BY'
LBL_INDICATED_BY = 'INDICATED_BY'
LBL_RESULTS_IN = 'RESULTS_IN'


class FunctionalRelation(MBDRelation):
	weight = FloatProperty(default=1.)

	def check_errors(self) -> list[tuple['FunctionalRelation', str]]:
		raise NotImplementedError(
			f'Method \'check_errors\' has not been implemented for \'{self.__class__.__name__}\'.')

	def get_url(self) -> str:
		raise NotImplementedError(f'Method \'get_url\' has not been implemented for \'{self.__class__.__name__}\'.')


class RealizesRelation(FunctionalRelation):
	def check_errors(self) -> list[tuple[FunctionalRelation, str]]:
		errors = []
		start_node = self.start_node()
		end_node = self.end_node()
		if not isinstance(start_node, Hardware):
			errors.append((self, 'Start node must be Hardware.'))
		if not isinstance(end_node, Function):
			errors.append((self, 'End node must be Function.'))
		if start_node.get_net() != end_node.get_net():
			errors.append((self, 'Start and end nodes must be in the same cluster.'))
		return errors

	def get_url(self) -> str:
		return f'/hardware/{self.start_node().uid}/realizes/{self.end_node().uid}/'


class RequiredForRelation(FunctionalRelation):
	operating_modes = ArrayProperty(base_property=StringProperty(), default=[])

	def check_errors(self) -> list[tuple[FunctionalRelation, str]]:
		errors = []
		start_node = self.start_node()
		end_node = self.end_node()
		if not isinstance(start_node, Function):
			errors.append((self, 'Start node must be Function.'))
		if not isinstance(end_node, Function):
			errors.append((self, 'End node must be Function.'))
		if start_node.has_subfunctions:
			errors.append((self, 'Start node may not have subfunctions.'))
		if end_node.has_subfunctions:
			errors.append((self, 'End node may not have subfunctions.'))
		return errors

	def get_url(self) -> str:
		return f'/function/{self.start_node().uid}/requiredfor/{self.end_node().uid}/'


class AffectsRelation(FunctionalRelation):
	def check_errors(self) -> list[tuple[FunctionalRelation, str]]:
		errors = []
		start_node = self.start_node()
		end_node = self.end_node()
		if not isinstance(start_node, Hardware):
			errors.append((self, 'Start node must be Hardware.'))
		if not isinstance(end_node, Function):
			errors.append((self, 'End node must be Function.'))
		if start_node.get_net() == end_node.get_net():
			errors.append((self, 'Start and end node must be in different clusters.'))
		return errors

	def get_url(self) -> str:
		return f'/hardware/{self.start_node().uid}/affects/{self.end_node().uid}/'


class SubfunctionOfRelation(FunctionalRelation):
	def check_errors(self) -> list[tuple[FunctionalRelation, str]]:
		errors = []
		start_node = self.start_node()
		end_node = self.end_node()
		if not isinstance(start_node, Function):
			errors.append((self, 'Start node must be Function.'))
		if not isinstance(end_node, Function):
			errors.append((self, 'End node must be Function.'))
		if start_node.get_net().get_net() != end_node.get_net():
			errors.append((self, 'Invalid functional breakdown through different layers.'))
		return errors

	def get_url(self) -> str:
		return f'/function/{self.end_node().uid}/subfunction/{self.start_node().uid}/'


class ObservedByRelation(FunctionalRelation):
	pass


class IndicatedByRelation(FunctionalRelation):
	pass


class ResultsInRelation(FunctionalRelation):
	pass


class FunctionalNode(MBDNode):
	_managed_relations: tuple[str] = ()
	RELATION_ATTRIBUTES: list[tuple[str]] = []

	net = RelationshipTo('mbdlyb.functional.gdb.Cluster', LBL_PARTOF, model=PartOfRelation, cardinality=ZeroOrOne)

	def check_errors(self) -> list[tuple[FunctionalRelation, str]]:
		errors: list[tuple[FunctionalRelation, str]] = []
		for r in self._managed_relations:
			for related_object in self.__getattribute__(r).all():
				errors += self.__getattribute__(r).relationship(related_object).check_errors()
		return errors


class OperatingMode(FunctionalNode):
	operating_modes = ArrayProperty(StringProperty(), required=True)

	def on_delete(self):
		updates = {opm: None for opm in self.operating_modes}
		self.update_relations_with_operating_mode(updates)

	@staticmethod
	def update_relations_with_operating_mode(updates: dict[str,Optional[str]]):
		current_opm_modes = ','.join([f'"{opm}"' for opm in updates.keys()])
		functions, _ = db.cypher_query(f'''MATCH(n:Function)-[r:REQUIRED_FOR]->(m:Function)
										WHERE any(x IN r.operating_modes WHERE x IN [{current_opm_modes}])
										RETURN n,m''',
									resolve_objects=True)
		for (start_node, end_node) in functions:
			rel: RequiredForRelation = start_node.__getattribute__('required_for').relationship(end_node)
			rel.operating_modes = [updates.get(opm, opm) for opm in rel.operating_modes]
			rel.operating_modes = [opm for opm in rel.operating_modes if opm is not None]
			rel.save()


class Observed(FunctionalNode):
	observed_by = RelationshipTo('mbdlyb.functional.gdb.DirectObservable', LBL_OBSERVED, model=ObservedByRelation)
	indicated_by = RelationshipTo('mbdlyb.functional.gdb.DiagnosticTestResult', LBL_INDICATED_BY, model=IndicatedByRelation)


class Function(Observed):
	_managed_relations = ('subfunctions', 'required_for')
	RELATION_ATTRIBUTES = Observed.RELATION_ATTRIBUTES + [
		('subfunction_of', 'subfunctions'), ('subfunctions', 'subfunction_of'), ('realized_by', 'realizes'),
		('affected_by', 'affects'), ('required_for', 'requires'), ('requires', 'required_for'),
		('observed_by', 'observed_functions'), ('indicated_by', 'indicated_functions')]

	realized_by = RelationshipFrom('mbdlyb.functional.gdb.Hardware', LBL_REALIZES, model=RealizesRelation)
	affected_by = RelationshipFrom('mbdlyb.functional.gdb.Hardware', LBL_AFFECTS, model=AffectsRelation)
	subfunction_of = RelationshipTo('mbdlyb.functional.gdb.Function', LBL_SUBFUNCTION, model=SubfunctionOfRelation)
	subfunctions = RelationshipFrom('mbdlyb.functional.gdb.Function', LBL_SUBFUNCTION, model=SubfunctionOfRelation)
	required_for = RelationshipTo('mbdlyb.functional.gdb.Function', LBL_REQUIRED, model=RequiredForRelation)
	requires = RelationshipFrom('mbdlyb.functional.gdb.Function', LBL_REQUIRED, model=RequiredForRelation)

	@property
	def has_subfunctions(self) -> bool:
		return len(self.subfunctions) > 0


class Hardware(Observed):
	_managed_relations = ('realizes', 'affects')
	BASE_PRIOR = {'Broken': 0.01}
	RELATION_ATTRIBUTES = Observed.RELATION_ATTRIBUTES + [('realizes', 'realized_by'), ('affects', 'affected_by'),
														  ('observed_by', 'observed_hardware'),
														  ('indicated_by', 'indicated_hardware')]

	fault_rates = JSONProperty(default=BASE_PRIOR)
	realizes = RelationshipTo(Function, LBL_REALIZES, model=RealizesRelation)
	affects = RelationshipTo(Function, LBL_AFFECTS, model=AffectsRelation)


class ObservableNode(FunctionalNode):
	def get_observed_nodes(self) -> list[FunctionalNode]:
		raise NotImplementedError(
			f'Method \'get_observed_nodes\' is not implemented for \'{self.__class__.__name__}\'.')

	def reposition(self) -> Optional[tuple[str, str, str]]:
		obs_nodes = self.get_observed_nodes()
		lfqn = longest_common_fqn(*obs_nodes)
		n_lfqn = self.root.get_node(lfqn)
		if not isinstance(n_lfqn, Cluster):
			n_lfqn: Cluster = n_lfqn.get_net()
		if self.get_net() == n_lfqn:
			return None
		repositioning = (self.fqn, self.get_net().fqn, n_lfqn.fqn)
		self.set_net(n_lfqn)
		return repositioning


class DirectObservable(ObservableNode):
	RELATION_ATTRIBUTES = ObservableNode.RELATION_ATTRIBUTES + [('observed_functions', 'observed_by'),
																('observed_hardware', 'observed_by')]

	observed_functions = RelationshipFrom('mbdlyb.functional.gdb.Function', LBL_OBSERVED, model=ObservedByRelation)
	observed_hardware = RelationshipFrom('mbdlyb.functional.gdb.Hardware', LBL_OBSERVED, model=ObservedByRelation)

	fp_rate = FloatProperty(default=.001)
	fn_rate = FloatProperty(default=.01)

	def get_observed_nodes(self) -> list[FunctionalNode]:
		return self.observed_functions.all() + self.observed_hardware.all()


class DiagnosticTest(ObservableNode):
	RELATION_ATTRIBUTES = ObservableNode.RELATION_ATTRIBUTES + [('test_results', 'results_from')]

	test_results = RelationshipTo('mbdlyb.functional.gdb.DiagnosticTestResult', LBL_RESULTS_IN, model=ResultsInRelation)

	fixed_cost = JSONProperty(default=dict())
	preconditions = JSONProperty(default=dict())

	def delete(self):
		for r in self.test_results:
			r.delete()
		super().delete()


class DiagnosticTestResult(ObservableNode):
	RELATION_ATTRIBUTES = ObservableNode.RELATION_ATTRIBUTES + [('results_from', 'results_from'),
																('indicated_functions', 'indicated_by'),
																('indicated_hardware', 'indicated_by')]
	results_from = RelationshipFrom('mbdlyb.functional.gdb.DiagnosticTest', LBL_RESULTS_IN, model=ResultsInRelation)
	indicated_functions = RelationshipFrom('mbdlyb.functional.gdb.Function', LBL_INDICATED_BY, model=IndicatedByRelation)
	indicated_hardware = RelationshipFrom('mbdlyb.functional.gdb.Hardware', LBL_INDICATED_BY, model=IndicatedByRelation)

	fp_rate = FloatProperty(default=.0001)
	fn_rate = FloatProperty(default=.001)

	def get_observed_nodes(self) -> list[FunctionalNode]:
		return self.indicated_functions.all() + self.indicated_hardware.all()


class Cluster(MBDNet):
	_managed_relations: tuple[str] = ('subnets', 'functions', 'hardware')
	_RELATION_ATTRS: dict[Type[MBDNode], dict[Type[MBDNode], str]] = {
		Hardware: {
			Function: 'realizes',
			DirectObservable: 'observed_by'
		},
		Function: {
			Function: 'required_for',
			DirectObservable: 'observed_by'
		}
	}
	XLS_NODE_CLASSES: dict[str, tuple[Type[FunctionalNode], tuple]] = {
		'Function': (Function, {}),
		'Hardware': (Hardware, {'fault_rates': Hardware.BASE_PRIOR}),
		'Direct': (DirectObservable, {'fp_rate': .001, 'fn_rate': .01}),
		'DiagnosticTest': (DiagnosticTest, {'fp_rate': .0001, 'fn_rate': .001})
	}
	GDB_NODE_TRANSFORM_CLASSES: dict[Type[FunctionalNode], tuple[Type[fn.FunctionalNode], list[str]]] = {
		Function: (fn.Function, []),
		Hardware: (fn.Hardware, ['fault_rates']),
		DirectObservable: (fn.DirectObservable, ['fp_rate', 'fn_rate']),
		DiagnosticTest: (fn.DiagnosticTest, ['fixed_cost', 'preconditions']),
		DiagnosticTestResult: (fn.DiagnosticTestResult, ['fp_rate', 'fn_rate']),
		OperatingMode: (fn.OperatingMode, ['operating_modes'])
	}
	GDB_RELATION_TRANSFORM_CLASSES: dict[Type[FunctionalRelation], tuple[Type[fn.FunctionalRelation], list[str]]] = {
		SubfunctionOfRelation: (fn.SubfunctionOfRelation, []),
		RequiredForRelation: (fn.RequiredForRelation, ['operating_modes']),
		RealizesRelation: (fn.RealizesRelation, []),
		AffectsRelation: (fn.AffectsRelation, []),
		ObservedByRelation: (fn.ObservedByRelation, []),
		ResultsInRelation: (fn.ResultsInRelation, []),
		IndicatedByRelation: (fn.IndicatedByRelation, [])
	}

	net = RelationshipTo('mbdlyb.functional.gdb.Cluster', LBL_PARTOF, model=PartOfRelation)
	subnets = RelationshipFrom('mbdlyb.functional.gdb.Cluster', LBL_PARTOF, model=PartOfRelation)
	functions = RelationshipFrom('mbdlyb.functional.gdb.Function', LBL_PARTOF, model=PartOfRelation)
	hardware = RelationshipFrom('mbdlyb.functional.gdb.Hardware', LBL_PARTOF, model=PartOfRelation)
	observables = RelationshipFrom('mbdlyb.functional.gdb.DirectObservable', LBL_PARTOF, model=PartOfRelation)
	tests = RelationshipFrom('mbdlyb.functional.gdb.DiagnosticTest', LBL_PARTOF, model=PartOfRelation)
	test_results = RelationshipFrom('mbdlyb.functional.gdb.DiagnosticTestResult', LBL_PARTOF, model=PartOfRelation)
	operating_modes = RelationshipFrom('mbdlyb.functional.gdb.OperatingMode', LBL_PARTOF, model=PartOfRelation)

	def get_subnets(self) -> dict[str, 'Cluster']:
		return {subnet.name: subnet for subnet in self.subnets.all()}

	def delete(self):
		for e in self.elements.all():
			e.delete()
		super().delete()

	@property
	def is_flat(self) -> bool:
		return len(self.subnets) == 0

	@property
	def is_abstract(self) -> bool:
		return all(not n.has_subfunctions for n in self.get_functions()) and all(
			n.is_abstract for n in self.get_subnets())

	def check_errors(self) -> list[tuple[FunctionalRelation, str]]:
		errors: list[tuple[FunctionalRelation, str]] = []
		for r in self._managed_relations:
			for related_object in self.__getattribute__(r).all():
				errors += related_object.check_errors()
		return errors

	@classmethod
	def load(cls, uid: str) -> fn.Cluster:
		def _retrieve_information() -> tuple[
			list[tuple[Cluster, Cluster, int]], dict[str, list[FunctionalNode]], dict[str, list[FunctionalRelation]]]:
			qr, _ = db.cypher_query(f'''CALL () {{
				MATCH p = (:Cluster {{uid: '{uid}'}})<-[:PART_OF*0..]-(c:Cluster)
				OPTIONAL MATCH (c)<-[:PART_OF]-(n WHERE NOT n:Cluster)
				OPTIONAL MATCH (pc:Cluster)<-[:PART_OF]-(c)
				RETURN c, pc, length(p) AS path_length, collect(n) AS c_nodes
			}}
			OPTIONAL MATCH (c)-[:PART_OF]->(pc:Cluster)
			RETURN c AS cluster, pc AS parent, path_length, c_nodes, 
				COLLECT {{MATCH (n WHERE n IN c_nodes)-[r WHERE type(r) <> 'PART_OF']->(m) RETURN r}} as relations
			ORDER BY path_length''',
									resolve_objects=True)
			cluster_data: list[tuple[Cluster, Cluster, int]] = []
			node_data: dict[str, list[FunctionalNode]] = dict()
			relation_data: dict[str, list[FunctionalNode]] = dict()
			for r in qr:
				cluster, parent_cluster, depth, nodes, relations = tuple(r)
				cluster_data.append((cluster, parent_cluster, depth))
				node_data[cluster.fqn] = nodes[0]
				relation_data[cluster.fqn] = relations[0]
			return cluster_data, node_data, relation_data

		def _adapt_fqn(root_fqn: str, fqn: str) -> str:
			return re.sub(f'^{root_fqn}(\\.|$)', '', fqn)

		def _create_cluster(cluster_data: list[tuple[Cluster, Cluster, int]],
							node_data: dict[str, list[FunctionalNode]],
							relation_data: dict[str, list[FunctionalRelation]]) -> fn.Cluster:
			db_root, _, db_root_depth = cluster_data[0]
			assert db_root_depth == 0, 'DB root cluster depth must be 0!'
			db_root_fqn = '' if db_root.at_root else db_root.get_net().fqn
			root: fn.Cluster = fn.Cluster(db_root.name)
			for c, pc, depth in cluster_data[1:]:
				parent_cluster = root if depth == 1 and pc.name == root.name else root.get_node(
					_adapt_fqn(db_root_fqn, pc.fqn))
				fn.Cluster(c.name, parent_cluster)
			for cluster_fqn, nodes in node_data.items():
				try:
					parent_cluster_fqn = _adapt_fqn(db_root_fqn, cluster_fqn)
					parent_cluster = root.get_node(parent_cluster_fqn)
					if not isinstance(parent_cluster, fn.Cluster):
						warnings.warn(f'Presumed cluster {parent_cluster_fqn} is not a cluster, skipping!')
						continue
					for node in nodes:
						klass, args = cls.GDB_NODE_TRANSFORM_CLASSES.get(type(node), (None, []))
						if klass is None:
							warnings.warn(f'Class of {node.fqn} could not be determined ({type(node)}), skipping!')
							continue
						argvals = [node.__getattribute__(arg) for arg in args]
						klass(node.name, *argvals, parent_cluster)
				except KeyError as e:
					warnings.warn(f'Cluster {cluster_fqn} could not be found, skipping!')
			operating_modes = root.get_all_opm_nodes()
			has_operating_modes = bool(operating_modes)
			value_opm_map: dict[str, OpmSet] = dict()
			for opm in operating_modes.keys():
				for v in opm.variable_names:
					value_opm_map[v] = opm
			for cluster_fqn, relations in relation_data.items():
				for relation in relations:
					try:
						source: fn.FunctionalNode = root.get_node(_adapt_fqn(db_root_fqn, relation.start_node().fqn))
						target: fn.FunctionalNode = root.get_node(_adapt_fqn(db_root_fqn, relation.end_node().fqn))
						klass, args = cls.GDB_RELATION_TRANSFORM_CLASSES.get(type(relation), (None, []))
						if klass is None:
							warnings.warn(f'Class of {relation} could not be determined, skipping!')
							continue
						argvals = [relation.__getattribute__(arg) for arg in args]
						if 'operating_modes' in args:
							idx = args.index('operating_modes')
							if has_operating_modes and relation.operating_modes:
								opm_set = value_opm_map[argvals[idx][0]]
								opm_node = operating_modes[opm_set]
								opm_value = OpmLogic.create(argvals[idx], opm_set)
								target.set_operating_modes(opm_set)
								if opm_node not in target.parents:
									fn.SelectOperatingModeRelation(opm_node, target)
							else:
								opm_value = OpmLogic()
							argvals.remove(argvals[idx])
							argvals.insert(idx, opm_value)
						klass(source, target, *argvals, relation.weight)
					except KeyError:
						warnings.warn(f'Failed to find one or more nodes in {relation}, skipping!')
			if has_operating_modes:
				root.propagate_operating_modes()
			return root

		return _create_cluster(*_retrieve_information())

	@classmethod
	def load_capella(
		cls,
		architecture: str,
		default_fn_tests: bool,
		default_hl_fn_tests: bool,
		default_hw_tests: bool,
		default_ll_hw_tests: bool,
		import_opms: bool,
		*files: Path) -> 'Cluster':
		from mbdlyb.functional.capella_import import CapellaToCluster  # lazy import
		aird_file = next((f for f in files if f.suffix == ".aird"), None)
		transformer = CapellaToCluster(aird_file)
		cluster = transformer.transform(
			architecture,
			default_fn_tests=default_fn_tests,
			default_hl_fn_tests=default_hl_fn_tests,
			default_hw_tests=default_hw_tests,
			default_ll_hw_tests=default_ll_hw_tests,
			import_opms=import_opms,
		)
		return cluster

	@classmethod
	def load_xls(cls, main: str, *files: Path, sheets: list[str] = None) -> 'Cluster':
		subnets: dict[str, 'Cluster'] = dict()
		connections: dict = dict()
		for file in files:
			if file.suffix == '.xlsx':
				_subnets = cls._load_xls(file, sheets)
				subnets.update(_subnets)
			elif file.suffix == '.json':
				if not file.exists():
					print(f'File {file.name} does not exist, skipping...')
					continue
				connections = cls._load_json(file)
		cls._substitute_nets(subnets)

		assert main in subnets, f'{main} could not be found in loaded networks!'
		net = subnets[main]
		update_fqn(net)
		net._update_operating_modes()

		for connection in connections:
			original_start_node_fqn = connection.get("original_source")
			original_end_node_fqn = connection.get("original_target")

			try:
				original_start_node = net.get_node(original_start_node_fqn)
				original_end_node = net.get_node(original_end_node_fqn)
			except KeyError:
				continue

			# remove original connection
			relation_attr = cls._RELATION_ATTRS.get(original_start_node.__class__).get(original_end_node.__class__)
			original_start_node.__getattribute__(relation_attr).disconnect(original_end_node)

			# add resolved connections
			for resolved_connection in connection.get("resolved_connections", dict()):
				start_node_fqn = resolved_connection.get("source")
				end_node_fqns = resolved_connection.get("targets")

				try:
					start_node = net.get_node(start_node_fqn)
				except KeyError:
					continue

				for fqn in end_node_fqns:
					try:
						end_node = net.get_node(fqn)
					except KeyError:
						continue
					relation_attr = cls._RELATION_ATTRS.get(start_node.__class__).get(end_node.__class__)
					start_node.__getattribute__(relation_attr).connect(end_node)

		return net

	@classmethod
	def _load_xls(cls, xls_file: Path, sheets: list[str] = None) -> dict[str, 'Cluster']:
		xls: dict[str, pd.DataFrame] = pd.read_excel(xls_file, sheet_name=None, header=[0, 1, 2], index_col=[0, 1, 2])
		sheets = sheets or list(xls.keys())
		subnets: dict[str, 'Cluster'] = {
			s_name: cls._sheet_to_fn_net(s_name, sheet.sort_index().sort_index(axis=1)) for s_name, sheet in
			xls.items() if s_name in sheets}
		cls._substitute_nets(subnets)
		return subnets

	@classmethod
	def _sheet_to_fn_net(cls, name: str, sheet: pd.DataFrame) -> 'Cluster':
		clusters = set(sheet.index.get_level_values(0)).difference({'Cost'})
		nodes = {c_name: cls._xls_to_cluster_net(c_name, sheet.loc[c_name]['Functions', c_name]) for c_name in clusters}
		net = Cluster(name=name).save()
		update_fqn(net)
		if name in sheet['Functions'].columns:
			nodes.update({f_name: Function(name=f_name).save() for f_name in sheet['Functions', name].columns})
			for (c, _, n), targets in sheet[sheet['Functions', name].any(axis=1)]['Functions', name].iterrows():
				for target, weight in targets[targets.notna()].items():
					weight, operating_modes = cls._parse_weight(weight)
					if len(operating_modes) > 0:
						raise RuntimeError('Operating modes are not allowed in the main function(s) of a cluster.')
					nodes[c].get_node(n).subfunction_of.connect(nodes[target], {'weight': weight})
		for node in nodes.values():
			node.set_net(net)
		for (source_cluster, source_type, source_name), targets in sheet['Functions'][list(clusters)].iterrows():
			for (target_cluster, target_name), weight in targets[targets.notna()].items():
				if source_cluster == target_cluster:
					continue
				weight, operating_modes = cls._parse_weight(weight)
				operating_modes = OpmLogic.create(operating_modes)
				_args = {'weight': weight}
				attr, _args = {
					'Function': (
						'required_for', {**_args, 'operating_modes': [v.name for v in operating_modes.variables]}),
					'Hardware': ('affects', _args)
				}.get(source_type, (None, None))
				if attr is None:
					raise RuntimeError(f'Cannot identify relation type for {source_type} to Function relation.')
				if attr == 'Hardware' and len(operating_modes) > 0:
					raise RuntimeError(
						'Operating modes are not allowed for inter-cluster hardware to function relations.')
				source_node = net.get_node(f'{source_cluster}.{source_name}')
				target_node = net.get_node(f'{target_cluster}.{target_name}')
				source_node.__getattribute__(attr).connect(target_node, _args)
		try:
			net._xls_add_observables(sheet[sheet['Observables'].any(axis=1)]['Observables'].droplevel(1))
		except KeyError:
			pass
		return net

	@classmethod
	def _xls_to_cluster_net(cls, name: str, sheet: pd.DataFrame) -> 'Cluster':
		nodes: dict[str, FunctionalNode] = dict()
		cluster = Cluster(name=name).save()
		update_fqn(cluster)
		for node_type, node_name in sheet.index:
			klass, kwargs = cls.XLS_NODE_CLASSES.get(node_type, (None, ()))
			if klass is None:
				raise RuntimeError(f'Incorrect type {node_type} provided for {name}.{node_name}!')
			nodes[node_name] = klass(name=node_name, **kwargs).save()
			nodes[node_name].set_net(cluster)
		for (_, source_name), targets in sheet[sheet.any(axis=1)].iterrows():
			for target_name, weight in targets[targets.notna()].items():
				relation_attr = cls._RELATION_ATTRS.get(nodes[source_name].__class__, dict()).get(
					nodes[target_name].__class__, None)
				if relation_attr is None:
					raise RuntimeError(
						f'Could not determine a compatible relation between {name}.{source_name} and {name}.{target_name}!')
				weight, operating_modes = cls._parse_weight(weight)
				_attrs = {'weight': weight}
				if relation_attr == 'required_for':
					operating_modes = OpmLogic.create(operating_modes)
					_attrs['operating_modes'] = [v.name for v in operating_modes.variables]
				nodes[source_name].__getattribute__(relation_attr).connect(nodes[target_name], _attrs)
		return cluster

	def _xls_add_observables(self, sheet: pd.DataFrame):
		try:
			costs = sheet.loc['Cost']
		except KeyError:
			costs = None
		else:
			sheet.drop('Cost', inplace=True)
		for (obs_type, obs_name), observed in sheet.T.iterrows():
			observed_weights = dict()
			for (cluster, name), weight in observed[observed.notna()].items():
				weight, test_results = self._parse_weight(weight)
				observed_weights[f'{cluster}.{name}'] = (weight, test_results)
			if not observed_weights:
				warnings.warn(f'Ignoring observable {obs_name} as it does not observe anything.')
				continue
			observable_net = self.get_node(longest_common_fqn(*observed_weights.keys()))
			if isinstance(observable_net, FunctionalNode):
				observable_net = observable_net.get_net()
			klass, kwargs = self.XLS_NODE_CLASSES.get(obs_type, None)
			if klass is None:
				raise RuntimeError(f'Unknown observable type {obs_type} defined for {obs_name}!')
			kwargs.update({'fixed_cost': dict(costs[obs_type, obs_name])}) if costs is not None and costs[
				obs_type, obs_name].any() else {}
			n = klass(name=obs_name, **kwargs).save()
			n.set_net(observable_net)
			test_result_names: dict[str, DiagnosticTestResult] = dict()
			for observed_fqn, (weight, test_results) in observed_weights.items():
				observed_node = observable_net.get_node(observable_net.rfqn(observed_fqn))
				relation_attr = self._RELATION_ATTRS.get(observed_node.__class__, dict()).get(klass, None)
				if relation_attr is not None:
					observed_node.__getattribute__(relation_attr).connect(n, {'weight': weight})

				# only for DiagnosticTests: add test-result nodes and their relations
				if isinstance(n, DiagnosticTest):
					# set default name for test-result node
					if len(test_results) == 0:
						test_results = [obs_name + '_result']
					# connect each result to the observed node ('observed_fqn')
					for result in test_results:
						# create the result-node if needed, connect it to the test ('obs_name')
						if result not in test_result_names:
							dt_result = DiagnosticTestResult(name=result).save()
							dt_result.set_net(observable_net)
							dt_result.results_from.connect(n, {'weight': weight})
							test_result_names[result] = dt_result
						# create relation between hardware / function and test-result
						observed_node.indicated_by.connect(test_result_names[result], {'weight': weight})

	def _update_operating_modes(self):
		all_opm_sets: list[tuple[OpmSet, list[Function]]] = []
		qr, _ = db.cypher_query(
			f'''MATCH (c:Cluster {{uid: '{self.uid}'}})(()<-[:PART_OF]-()){{0,99}}(f:Function)<-[r:REQUIRED_FOR WHERE size(r.operating_modes) > 0]-()
RETURN f, apoc.coll.flatten(collect(r.operating_modes))''',
			resolve_objects=True)
		for r in qr:
			node, operating_modes = tuple(r)
			all_opm_variables = [OpmVariable(opm_label) for opm_label in operating_modes[0]]
			opm_sets = [(s, n) for s, n in all_opm_sets if any([s.contains_variable(v) for v in all_opm_variables])]

			# check if this node is directly affected by operating modes
			if len(all_opm_variables) == 0:
				continue

			# create a new set if none is found, otherwise use the one found
			if len(opm_sets) == 0:
				opm_set = OpmSet()
				node_list = []
				all_opm_sets.append((opm_set, node_list))
			elif len(opm_sets) == 1:
				opm_set, node_list = opm_sets[0]
			else:
				raise RuntimeError('Too many sets of operating modes found.')
			opm_set.add_variables(all_opm_variables)
			node_list.append(node)

		# create a select node for each set of operating modes
		for opm_set, nodes in all_opm_sets:
			new_name = 'Select_' + '_'.join(opm_set.variable_names)
			opm_node = OperatingMode(name=new_name, operating_modes=opm_set.variable_names).save()
			lcfqn = longest_common_fqn(*[n.fqn for n in nodes])
			parent = self.get_node(lcfqn)
			if not isinstance(parent, Cluster):
				parent = parent.get_net()
			opm_node.set_net(parent)

	@staticmethod
	def _parse_weight(weight: str) -> tuple[float, list[str]]:
		# return the weight as a float if it specified as a number
		if isinstance(weight, (int, float)):
			return float(weight), list()
		# return a weight of 1.0 if it is marked with a single character, e.g. 'X'
		if len(weight) == 1:
			return 1.0, []
		# check if a weight is given after the operating modes
		if ':' in weight:
			weight, weight_value = weight.split(':', 1)
		else:
			weight_value = 1.0
		weight_value = float(weight_value)
		# parse the definition of operating modes
		modes: list[str] = list()
		for mode in weight.split(','):
			modes.append(mode.strip())
		return weight_value, modes

	@classmethod
	def _substitute_nets(cls, subnets: dict[str, 'Cluster']):
		subnet_names = list(subnets.keys())
		for sn_name in subnet_names:
			if any(sn._substitute_net(subnets[sn_name]) for sn in subnets.values() if sn.name != sn_name):
				subnets.pop(sn_name)

	def _substitute_net(self, subnet: 'Cluster') -> bool:
		b: bool = any(sn._substitute_net(subnet) for sn in self.subnets.all())
		if subnet.name in self.get_subnets():
			self.get_subnets()[subnet.name]._substitute_with(subnet)
			b = True
		return b

	def _substitute_with(self, subnet: 'Cluster'):
		my_functions = {n.name for n in self.functions.all()}
		subnet_functions = {n.name for n in subnet.functions.all()}
		if my_functions.symmetric_difference(subnet_functions):
			raise RuntimeError(
				f'Functions of substitution network for {self.name} do not match: {my_functions} != {subnet_functions}!')
		node_names_to_substitute = {n.name for n in self.mbdnodes.all()}.intersection(
			{n.name for n in subnet.mbdnodes.all()})
		for node_name in node_names_to_substitute:
			x_node = subnet.elements.get(name=node_name)
			for x_attr, y_attr in type(x_node).RELATION_ATTRIBUTES:
				for y_node in x_node.__getattribute__(x_attr).all():
					y_node.__getattribute__(y_attr).reconnect(x_node, self.get_node(node_name))
			x_node.delete()
		for e in subnet.elements.all():
			e.set_net(self)
		subnet.delete()

	@classmethod
	def _load_json(cls, f: Path) -> None:
		with f.open('r') as fr:
			contents = json.load(fr)
			return contents.get("connections", list())


def update_fqn(obj: MBDElement):
	obj.fqn = obj.get_fqn()
	obj.save()
	if isinstance(obj, MBDNet):
		for element in obj.elements.all():
			update_fqn(element)


def update_name(obj):
	net = obj.get_net()
	if net:
		sibling_names = {node.name for node in net.elements if node.uid != obj.uid}
	else:
		sibling_names = {node.name for node in Cluster.nodes.has(net=False)}

	basename = obj.name
	name = basename
	counter = 1
	while name in sibling_names:
		name = f"{basename}_{counter}"
		counter += 1

	obj.name = name
	update_fqn(obj)
