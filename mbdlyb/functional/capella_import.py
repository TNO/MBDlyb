# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import re
from typing import Union

from capellambse import MelodyModel
from capellambse.metamodel import cs, fa

from mbdlyb import longest_common_fqn
from mbdlyb.functional.gdb import (Cluster, DiagnosticTest, DiagnosticTestResult, Function, Hardware, OperatingMode,
								   RequiredForRelation, DirectObservable, update_fqn)
from mbdlyb.ui.helpers import get_object_or_404, get_relation_or_404


class CapellaToCluster:
	"""Class for converting Capella model to a Cluster object as defined
	in MBDlyb.
	"""

	_aird: str = None
	_model: MelodyModel = None
	_archictecture = None
	_archictecture_root = None
	_arch_dict: dict = None
	_cluster: Cluster = None
	_elements: dict = None

	_observe_open_ports: bool = True
	_default_hw_tests: bool = False
	_default_ll_hw_tests: bool = False
	_default_fn_tests: bool = False
	_default_hl_fn_tests: bool = False
	_import_opms: bool = False

	_diagnostic_test_mapping: dict[DiagnosticTestResult, list[str]]
	_functions_to_position: list[Function]

	def __init__(self, aird: str) -> None:
		self._aird = aird
		self._model = self._load_model()
		self._arch_dict = {"logical": self._model.la, "physical": self._model.pa}
		self._diagnostic_test_mapping = dict()
		self._functions_to_position = []

	def transform(
			self,
			architecture: str,
			default_fn_tests: bool = False,
			default_hl_fn_tests: bool = False,
			default_hw_tests: bool = False,
			default_ll_hw_tests: bool = False,
			import_opms: bool = False,
	) -> Cluster:
		"""Transforms Capella architecture to Neomodel Cluster."""
		self._set_architecture(architecture)
		self._default_fn_tests = default_fn_tests
		self._default_hl_fn_tests = default_hl_fn_tests
		self._default_hw_tests = default_hw_tests
		self._default_ll_hw_tests = default_ll_hw_tests
		self._import_opms = import_opms

		# format names of Capella objects
		capella_objects = [
			*self._archictecture.all_components,
			*self._archictecture.all_functions,
		]
		for obj in capella_objects:
			obj.name = self._format(obj.name)

		# add Cluster, Function, and Hardware nodes to Cluster
		self.cluster = self._build_net(self._archictecture_root, True)
		update_fqn(self.cluster)

		# add Function-Function and Hardware-Function relations
		for fn in self._archictecture.all_functions:
			self._add_subfunction_of_relations(fn)
		self._position_remaining_functions()
		update_fqn(self.cluster)

		self._connect_test_results()
		self._add_required_for_relations()

		if self._import_opms:
			self._add_operating_modes()

		# add default tests
		if self._default_fn_tests:
			for n in self._elements[Function].values():
				self._add_default_diagnostic_test(n)
		elif self._default_hl_fn_tests:
			for n in self._elements[Function].values():
				if n.has_subfunctions:
					self._add_default_diagnostic_test(n)

		if self._default_hw_tests:
			for n in self._elements[Hardware].values():
				self._add_default_diagnostic_test(n)
		elif self._default_ll_hw_tests:
			for n in self._elements[Hardware].values():
				if not any(f.has_subfunctions for f in n.realizes.all()):
					self._add_default_diagnostic_test(n)

		# ensure unique FQNs
		self._format_fqns()
		return self.cluster

	def _load_model(self) -> MelodyModel:
		"""Loads Capella model using capellambse library."""
		return MelodyModel(self._aird)

	def _reset(self) -> None:
		"""Resets node dictionary with UUIDs of Capella objects and their
		FQN in MBD Cluster as key, value pairs.
		"""
		self._elements = {n: dict()
						  for n in [Cluster, Hardware, Function, RequiredForRelation]}
		self._cluster = None

	def _set_architecture(self, arch: str) -> None:
		"""Sets architecture of Capella model to transform.
		Can be 'la' (logical) or 'pa' (physical).
		"""
		self._reset()
		self._archictecture = self._arch_dict.get(arch)
		self._archictecture_root = self._set_root()

	@staticmethod
	def _format(s: str) -> str:
		"""Removes leading and trailing whitespace. Replaces
		full stops (.) and whitespace with underscores (_).
		"""
		return re.sub(r'\s+', '_', s.strip()).replace('.', '_')

	@staticmethod
	def _fn_is_diagnostic_test(capella_fn: fa.Function) -> bool:
		return bool(capella_fn.applied_property_value_groups) and \
			capella_fn.property_value_groups['MBDlyb.Diagnostic test']['Diagnostic test']

	@staticmethod
	def _hw_is_inspectable(capella_hw: cs.Component) -> bool:
		return bool(capella_hw.applied_property_value_groups) and \
			capella_hw.property_value_groups['MBDlyb.Inspections']['Inspectable']

	@staticmethod
	def _fn_root_of_diagnostic_tree(capella_fn: fa.Function) -> bool:
		if CapellaToCluster._fn_is_diagnostic_test(capella_fn):
			return True
		if capella_fn.is_leaf:
			return False
		return all(CapellaToCluster._fn_root_of_diagnostic_tree(f) for f in capella_fn.functions)

	def _get_elements_cluster(
			self, cluster: Cluster
	) -> list[Union[Cluster, Hardware, Function]]:
		"""Returns all MBDElement objects in a Cluster object,
		including the Cluster itself.
		"""
		elements = [
			_n
			for n in cluster.elements
			for _n in (self._get_elements_cluster(n) if isinstance(n, Cluster) else [n])
		]
		return [*elements, cluster]

	def _set_root(self) -> cs.Component:
		"""Selects root component. Handles cases with multiple
		root components (not supported by capellambse library).
		"""
		roots = [
			r
			for r in self._archictecture.component_package.components
			if not r.is_actor
		]
		if len(roots) > 1:
			print("Multiple root components found. Please select one:")
			for i, comp in enumerate(roots, start=1):
				print(f"{i}: {comp.name}")
			selected = int(input("Enter number: "))
			return roots[selected - 1]
		return roots[0]

	def _format_fqns(self) -> None:
		"""Ensures uniqueness of all FQNs in Cluster. Capella does not require
		unique names for each object.
		"""
		for el in self._get_elements_cluster(self.cluster):
			net = el.get_net()
			if net:
				names_in_net = [n.name for n in net.elements if n.uid != el.uid]
				count = 0
				new_name = el.name
				while new_name in names_in_net:
					count += 1
					new_name = f"{el.name}_{count}"
				el.name = new_name
				el.save()
		update_fqn(self.cluster)

	def _add_node(
			self,
			capella_obj: Union[cs.Component, fa.Function],
			mbd_el: Union[Cluster, Hardware, Function],
	) -> None:
		"""Adds node of Capella object to node dict."""
		el_type = type(mbd_el)
		if capella_obj.uuid not in self._elements[el_type]:
			self._elements[el_type][capella_obj.uuid] = mbd_el
		else:
			print(
				f"{el_type} with Capella UUID {capella_obj.uuid!r} already in network."
			)

	def _add_required_for_relations(self) -> None:
		"""Adds functional exchanges in Capella as required_for relations
		in Cluster.
		"""
		capella_fn_exchanges = []
		for exch in self._archictecture.all_function_exchanges:
			src_uuid = exch.source.owner.uuid
			trgt_uuid = exch.target.owner.uuid

			if (
					src_uuid in self._elements[Function]
					and trgt_uuid in self._elements[Function]
			):
				# only adds each relation from source to target once
				if (src_uuid, trgt_uuid) not in capella_fn_exchanges:
					capella_fn_exchanges.append((src_uuid, trgt_uuid, exch.uuid))

		for src_uuid, trgt_uuid, exch_uuid in capella_fn_exchanges:
			src_fn = self._elements[Function].get(src_uuid)
			trgt_fn = self._elements[Function].get(trgt_uuid)
			rel = trgt_fn.requires.connect(src_fn)
			self._elements[RequiredForRelation][exch_uuid] = rel

	def _add_subfunction_of_relations(self, capella_fn: fa.Function) -> None:
		"""Adds function breakdown in Capella as subfunction_of relations
		in Cluster.
		"""
		parent = capella_fn.parent
		if capella_fn.uuid in self._elements[Function] and isinstance(
				parent, fa.Function
		) and not self._fn_root_of_diagnostic_tree(parent):
			fn = self._elements[Function].get(capella_fn.uuid)

			# if parent function not in Cluster create Function node
			if parent.uuid not in self._elements[Function]:
				parent_fn = Function(name=parent.name).save()
				self._add_node(parent, parent_fn)
				self._functions_to_position.append(parent_fn)

			parent_fn = self._elements[Function].get(parent.uuid)

			fn.subfunction_of.connect(parent_fn)

	def _position_remaining_functions(self) -> None:
		"""Determine the correct cluster for putting in the functions
		not explicitly being part of one. These functions are usually
		derived from the functional breakdown."""
		for ftr in self._functions_to_position:
			if len(ftr.subfunctions) == 1:  # With only one subfunction, put at higher level
				cluster = ftr.subfunctions.single().get_net()
				if not cluster.at_root:
					cluster = cluster.get_net()
			else:  # With more subfunctions, determine the common level
				lfqn = longest_common_fqn(*[sf.fqn for sf in ftr.subfunctions])
				cluster = self.cluster.get_node(lfqn)
			ftr.set_net(cluster)

	def _connect_test_results(self) -> None:
		"""Connects the test results that have yet to be connected to the intended
		functions."""
		for dtr, f_uuids in self._diagnostic_test_mapping.items():
			for f_uuid in f_uuids:
				dtr.indicated_functions.connect(self._elements[Function][f_uuid])

	def _group_operating_modes(self, rel_to_chains: dict) -> None:
		"""Group functional chains with same target function to same
		operating mode"""
		nodes = dict()  # map function nodes to associated chains
		for rel, chains in rel_to_chains.items():
			nodes.setdefault(rel.start_node().uid, set()).update(chains)
			nodes.setdefault(rel.end_node().uid, set()).update(chains)

		opms_dict = dict()
		for rel in rel_to_chains:
			start_uid = rel.start_node().uid
			end_uid = rel.end_node().uid

			start_chains = nodes[start_uid]
			end_chains = nodes[end_uid]
			if len(end_chains) > len(start_chains) and start_chains != end_chains:
				# add or update target function if end node of relation is
				# part of different set of functional chains than start node
				opms_dict.setdefault(end_uid, []).append((start_uid, start_chains))
		return opms_dict

	def _add_operating_modes(self) -> None:
		"""Transforms functional chains in Capella to operating modes."""
		func_chains = self._archictecture.all_functional_chains
		uuid_chains = {chain.uuid: chain for chain in func_chains}

		# map 'required for' relations to associated functional chains
		rel_to_chains = dict()
		for chain in func_chains:
			for link in chain.involved_links:
				rel = self._elements[RequiredForRelation][link.uuid]
				rel_to_chains.setdefault(rel, set()).add(chain.uuid)

		opms_dict = self._group_operating_modes(rel_to_chains)
		for end_uid, start_nodes in opms_dict.items():
			_, start_chains = zip(*start_nodes)

			# add select node
			opm_names = sorted([
				self._format(uuid_chains[chain].name)
				for chains in start_chains for chain in chains
			])
			name = 'Select_' + '_'.join(opm_names)
			opm = OperatingMode(name=name, operating_modes=opm_names).save()
			end_node = get_object_or_404(Function, uid=end_uid)
			opm.set_net(end_node.get_net())

			# add operating mode to required for relations
			for (start_uid, start_chains) in start_nodes:
				start_node = get_object_or_404(Function, uid=start_uid)
				rel = get_relation_or_404(start_node, end_node, "required_for")
				rel.operating_modes.extend([
					self._format(uuid_chains[start_chain].name)
					for start_chain in start_chains
				])
				rel.save()

	def _add_default_diagnostic_test(self, mbd_el: Hardware | Function) -> None:
		"""Adds a DiagnosticTest and DiagnosticTestResult pair to a
		Hardware or Function node.
		"""
		assert isinstance(
			mbd_el, (Hardware, Function)
		), "Node must be of type Hardware or Function"

		result = DiagnosticTestResult(name="Test_" + mbd_el.name + "_Result").save()
		result.set_net(mbd_el.get_net())
		mbd_el.indicated_by.connect(result)

		test = DiagnosticTest(name="Test_" + mbd_el.name,
							  fixed_cost={'Time': 1.}).save()
		test.set_net(mbd_el.get_net())
		result.results_from.connect(test)

	def _add_specified_diagnostic_test(self, cluster: Cluster, name: str, cost: float) -> tuple[
		DiagnosticTest, DiagnosticTestResult]:
		dt = DiagnosticTest(name=name, fixed_cost={'Time': cost}).save()
		dt.set_net(cluster)
		dtr = DiagnosticTestResult(name=name + '_Result').save()
		dtr.set_net(cluster)
		dt.test_results.connect(dtr)
		return dt, dtr

	def _build_net(self, root: cs.Component, is_sys_root: bool = False) -> Cluster:
		"""Builds Cluster from Capella architecture."""
		cluster = Cluster(name=root.name).save()
		self._add_node(root, cluster)
		update_fqn(cluster)

		functions: list[Function] = []

		for capella_fn in root.allocated_functions:
			if self._fn_is_diagnostic_test(capella_fn):
				cost = capella_fn.property_value_groups['MBDlyb.Diagnostic test']['Cost']
				_, dtr = self._add_specified_diagnostic_test(
					cluster, capella_fn.name, cost)
				self._diagnostic_test_mapping[dtr] = [f.uuid for f in capella_fn.functions]
				continue

			fn = Function(name=capella_fn.name).save()
			fn.set_net(cluster)
			self._add_node(capella_fn, fn)
			functions.append(fn)

			for uo in [o for o in capella_fn.outputs if not o.exchanges]:
				if bool(uo.applied_property_value_groups) and uo.property_value_groups['MBDlyb.Direct observables']['Observable']:
					do = DirectObservable(name=self._format(uo.name)).save()
					do.set_net(cluster)
					do.observed_functions.connect(fn)

		if not is_sys_root and functions:
			hw = Hardware(name=root.name).save()
			hw.set_net(cluster)
			self._add_node(root, hw)
			for fn in functions:
				hw.realizes.connect(fn)

			if self._hw_is_inspectable(root):
				cost = root.property_value_groups['MBDlyb.Inspections']['Cost']
				_, dtr = self._add_specified_diagnostic_test(
					cluster, f'Inspect_{root.name}', cost)
				dtr.indicated_hardware.connect(hw)

		for capella_comp in root.components:
			if capella_comp.uuid not in self._elements[Cluster]:
				subcluster = self._build_net(capella_comp)
				subcluster.set_net(cluster)
		return cluster
