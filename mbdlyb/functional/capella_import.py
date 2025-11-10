# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import re
from multiprocessing import Queue
from typing import Union, Optional

from capellambse import MelodyModel
from capellambse.metamodel import cs, fa

from neomodel import db

from mbdlyb.functional.gdb import (Cluster, DiagnosticTest, DiagnosticTestResult, Function, Hardware, ObservedError,
								   OperatingMode, RequiredForRelation, DirectObservable, update_fqn)


class CapellaToCluster:
	"""Class for converting Capella model to a Cluster object as defined
	in MBDlyb.
	"""

	_aird: str = None
	_model: MelodyModel = None
	_architecture = None
	_architecture_root = None
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
	_function_positions: dict[str, tuple[Cluster, Hardware]]

	_capella_fns: dict[str, fa.Function]
	_error_exchanges: set[str]
	_initial_error_propagations: dict[str, dict[str, list[str]]]
	_non_self_error_reporting_functions: set[Function]
	_error_code_nodes: dict[str, ObservedError]

	_queue: Optional[Queue]

	def __init__(self, aird: str, queue: Queue = None) -> None:
		self._aird = aird
		self._model = self._load_model()
		self._arch_dict = {"logical": self._model.la, "physical": self._model.pa}
		self._diagnostic_test_mapping = dict()
		self._function_positions = dict()
		self._capella_fns = dict()
		self._error_exchanges = set()
		self._initial_error_propagations = dict()
		self._non_self_error_reporting_functions = set()
		self._error_code_nodes = dict()
		self._queue = queue

	def transform(
			self,
			architecture: str,
			default_fn_tests: bool = False,
			default_hl_fn_tests: bool = False,
			default_hw_tests: bool = False,
			default_ll_hw_tests: bool = False,
			import_opms: bool = False
	) -> Cluster:
		"""Transforms Capella architecture to Neomodel Cluster."""
		self._set_architecture(architecture)
		self._default_fn_tests = default_fn_tests
		self._default_hl_fn_tests = default_hl_fn_tests
		self._default_hw_tests = default_hw_tests
		self._default_ll_hw_tests = default_ll_hw_tests
		self._import_opms = import_opms

		self._update_status(0.)

		self._cluster = self._build_component_tree(self._architecture.root_component)
		self._update_status(.1)
		self._build_functional_tree(self._architecture.root_function)
		self._update_status(.2)

		self._connect_test_results()
		self._update_status(.3)
		self._position_remaining_nodes()
		self._update_status(.4)
		update_fqn(self._cluster)
		self._update_status(.5)

		self._add_error_propagation_relations()
		self._update_status(.6)
		self._add_required_for_relations()
		self._update_status(.7)
		self._add_observed_errors()
		self._update_status(.7)

		if self._import_opms:
			self._add_operating_modes()
		self._update_status(.8)

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
		self._update_status(.9)

		# ensure unique FQNs
		self._format_fqns()
		self._update_status(1.)
		return self._cluster

	def _update_status(self, status: float):
		if self._queue is not None:
			self._queue.put_nowait(status)

	def _build_component_tree(self, capella_comp: cs.Component) -> Optional[Cluster]:
		if not capella_comp.components and not capella_comp.allocated_functions:
			return None
		c = Cluster(name=self._format(capella_comp.name)).save()
		self._add_node(capella_comp, c)
		if capella_comp.allocated_functions:
			hw = self._add_hardware(capella_comp, c)
			for capella_fn in capella_comp.allocated_functions:
				self._function_positions[capella_fn.uuid] = (c, hw)
		children = [self._build_component_tree(child_comp) for child_comp in capella_comp.components]
		for subcluster in children:
			if subcluster is not None:
				subcluster.set_net(c)
		return c

	def _build_functional_tree(self, capella_fn: fa.Function, parent_dt: DiagnosticTest = None)\
				-> Optional[Function]:
		name = self._format(capella_fn.name)
		fn: Function = None
		if self._fn_is_diagnostic_test(capella_fn):
			if parent_dt is None:  # we have a top-level test
				cost = capella_fn.property_value_groups['MBDlyb.Diagnostic test']['Cost']
				dt, dtr = self._add_specified_diagnostic_test(self._format(capella_fn.name), cost)
				self._diagnostic_test_mapping[dtr] = [f.uuid for f in
													  list(capella_fn.functions) + [e.source.owner for i in
																					capella_fn.inputs for e in
																					i.exchanges]]
				for child in capella_fn.functions:
					self._build_functional_tree(child, dt)
				return None
			else:  # we have a subtest, we ignore the cost
				fn = self._add_function(capella_fn)
				dtr = DiagnosticTestResult(name=name + '_Result').save()
				parent_dt.test_results.connect(dtr)
				self._diagnostic_test_mapping[dtr] = [capella_fn.uuid]  # for now
		elif parent_dt is not None:  # subfunction of a test
			fn = self._add_function(capella_fn)
			# self._add_direct_observables_to_fn(capella_fn, fn)  # maybe later
		elif self._fn_root_of_diagnostic_tree(capella_fn):  # no function to be added here
			for child in capella_fn.functions:
				self._build_functional_tree(child, parent_dt)
			return None
		else:  # normal function
			fn = self._add_function(capella_fn)
			self._add_direct_observables_to_fn(capella_fn, fn)

		children = [self._build_functional_tree(child_fn, parent_dt) for child_fn in
					capella_fn.functions]
		for subfunction in children:
			if subfunction is not None:
				subfunction.subfunction_of.connect(fn)
		return fn

	def _add_hardware(self, capella_hw: cs.Component, cluster: Cluster) -> Optional[Hardware]:
		complete_composite = False
		if capella_hw.property_value_groups.get('MBDlyb.Composite'):
			complete_composite = capella_hw.property_value_groups['MBDlyb.Composite']['Complete composite']
		if complete_composite:
			return None

		fault_rate = 0.01  # default
		if capella_hw.property_value_groups.get('MBDlyb.Hardware priors'):
			fault_rate = capella_hw.property_value_groups['MBDlyb.Hardware priors']['Fault rate']

		hw = Hardware(name=self._format(capella_hw.name), fault_rates={'Broken': fault_rate}).save()
		hw.set_net(cluster)
		self._add_node(capella_hw, hw)
		if self._hw_is_inspectable(capella_hw):
			cost = capella_hw.property_value_groups['MBDlyb.Inspections']['Cost']
			_, dtr = self._add_specified_diagnostic_test(f'Inspect_{self._format(capella_hw.name)}', cost, cluster)
			dtr.indicated_hardware.connect(hw)
		return hw

	def _add_function(self, capella_fn: fa.Function) -> Function:
		self._capella_fns[capella_fn.uuid] = capella_fn
		fn = Function(name=self._format(capella_fn.name)).save()
		self._add_node(capella_fn, fn)
		if capella_fn.uuid in self._function_positions:
			c, hw = self._function_positions[capella_fn.uuid]
			fn.set_net(c)
			if hw is not None:
				fn.realized_by.connect(hw)
		return fn

	def _add_specified_diagnostic_test(self, name: str, cost: float, cluster: Cluster = None) -> tuple[
		DiagnosticTest, DiagnosticTestResult]:
		dt = DiagnosticTest(name=name, fixed_cost={'Time': cost}).save()
		dtr = DiagnosticTestResult(name=name + '_Result').save()
		dt.test_results.connect(dtr)
		if cluster is not None:
			dt.set_net(cluster)
			dtr.set_net(cluster)
		return dt, dtr

	def _add_direct_observables_to_fn(self, capella_fn: fa.Function, fn: Function):
		for o in [_o for _o in capella_fn.outputs if _o.applied_property_value_groups]:
			do = None
			if 'MBDlyb.Error reporting' in o.property_value_groups.keys():
				is_reporting = o.property_value_groups['MBDlyb.Error reporting']['Reporting errors']
				ec = o.property_value_groups['MBDlyb.Error reporting']['Error code']
				if is_reporting and bool(ec):
					do = ObservedError(name=self._format(o.name), error_code=ec).save()
					do.observed_function.connect(fn, {'error_code': ec})
				elif is_reporting:
					self._non_self_error_reporting_functions.add(fn)
				elif bool(ec):
					for ex in o.exchanges:
						self._error_exchanges.add(ex.uuid)
						if capella_fn.uuid not in self._initial_error_propagations:
							self._initial_error_propagations[capella_fn.uuid] = {ex.target.parent.uuid: [ec]}
						else:
							if ex.target.parent in self._initial_error_propagations[capella_fn.uuid]:
								self._initial_error_propagations[capella_fn.uuid][ex.target.parent.uuid].append(ec)
							else:
								self._initial_error_propagations[capella_fn.uuid][ex.target.parent.uuid] = [ec]

			if not o.exchanges:
				if 'MBDlyb.Direct observables' in o.property_value_groups.keys() and \
						o.property_value_groups['MBDlyb.Direct observables']['Observable']:
					do = DirectObservable(name=self._format(o.name)).save()
					do.observed_functions.connect(fn)
			if do is not None and capella_fn.uuid in self._function_positions:
				do.set_net(self._function_positions[capella_fn.uuid][0])

	def _add_error_propagation_relations(self):
		for source_uuid, target_data in self._initial_error_propagations.items():
			source_fn = self._elements[Function].get(source_uuid)
			for target_uuid, ecs in target_data.items():
				for ec in ecs:
					oe = ObservedError(name=self._format(ec), error_code=ec).save()
					oe.observed_function.connect(source_fn, {'error_code': ec})
					self._error_code_nodes[ec] = oe
				target_fn = self._elements[Function].get(target_uuid)
				source_fn.communicates_through.connect(target_fn, {'error_codes': ecs})
				self._propagate_error_relations(self._capella_fns[target_uuid], ecs)

	def _propagate_error_relations(self, cfn: fa.Function, ecs: list[str]):
		source_fn = self._elements[Function][cfn.uuid]
		for o in cfn.outputs:
			if o.exchanges:
				for e in o.exchanges:
					target = e.target.parent
					target_fn = self._elements[Function][target.uuid]
					if source_fn.required_for.is_connected(target_fn):
						rel = source_fn.required_for.relationship(target_fn)
						rel.error_codes += ecs
						rel.save()
					else:
						source_fn.required_for.connect(target_fn, {'error_codes': ecs})
						self._error_exchanges.add(e.uuid)
					self._propagate_error_relations(target, ecs)

	def _add_observed_errors(self):
		for of in self._non_self_error_reporting_functions:
			for predecessor in of.requires.all():
				for rel in of.requires.all_relationships(predecessor):
					for ec in rel.error_codes:
						# do = ObservedError(name=self._format(ec), error_code=ec).save()
						do = self._error_code_nodes[ec]
						do.reporting_function.connect(of, {'error_code': ec})
						if of.get_net() is not None:
							do.set_net(of.get_net())


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
		self._architecture = self._arch_dict.get(arch)
		self._architecture_root = self._set_root()

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
		return bool(capella_hw.property_value_groups.get('MBDlyb.Inspections')) and \
			capella_hw.property_value_groups['MBDlyb.Inspections']['Inspectable']

	@staticmethod
	def _exch_is_required(capella_exch: fa.FunctionalExchange) -> bool:
		""" Check whether the function exchange is required.
			This returns false only if the functional exchange has been explicitly labeled as not required.
		"""
		if not(bool(capella_exch.applied_property_value_groups)):
			return True  # default if there is no property value present
		else:
			return capella_exch.property_value_groups['MBDlyb.Required functional exchange']['Required']

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
			for r in self._architecture.component_package.components
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
		for el in self._get_elements_cluster(self._cluster):
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
		update_fqn(self._cluster)

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
		for exch in self._architecture.all_function_exchanges:
			if exch.uuid in self._error_exchanges:
				continue

			src_uuid = exch.source.parent.uuid
			trgt_uuid = exch.target.parent.uuid

			if (
					src_uuid in self._elements[Function]
					and trgt_uuid in self._elements[Function]
					and self._exch_is_required(exch)
			):
				# only adds each relation from source to target once
				if (src_uuid, trgt_uuid) not in capella_fn_exchanges:
					capella_fn_exchanges.append((src_uuid, trgt_uuid, exch.uuid))

		for src_uuid, trgt_uuid, exch_uuid in capella_fn_exchanges:
			src_fn = self._elements[Function].get(src_uuid)
			trgt_fn = self._elements[Function].get(trgt_uuid)
			rel = trgt_fn.requires.connect(src_fn)
			self._elements[RequiredForRelation][exch_uuid] = rel

	def _position_remaining_nodes(self) -> None:
		# Position unpositioned functions based on their subfunctions
		db.cypher_query('''CALL () {
    MATCH (root:Cluster)
    WHERE NOT (root)-[:PART_OF]->(:Cluster)
    MATCH (f:Function)
    WHERE NOT (f)-[:PART_OF]->(:Cluster)
    RETURN root, collect(f) AS dislocated_functions
}
WITH *
UNWIND dislocated_functions AS f
CALL (f, root) {
    MATCH (sf:Function)-[:SUBFUNCTION_OF*]->(f)
    MATCH (c:Cluster)<-[:PART_OF]-(sf)
    WITH collect(c) as cs, collect(sf) as sfs
    MATCH (cr:Cluster)
    WHERE size(cs) > 0 AND all(cx IN cs WHERE (cx)-[:PART_OF*]->(cr))
    MATCH p = (cr)-[:PART_OF*]->(root)
    RETURN cr, cs, p, length(p) AS length
    ORDER BY length(p) DESC
    LIMIT 1
}
CREATE (f)-[:PART_OF]->(cr)''')
		# Position unpositioned functions based on their superfunctions
		db.cypher_query('''CALL () {
    MATCH (root:Cluster)
    WHERE NOT (root)-[:PART_OF]->(:Cluster)
    MATCH (f:Function)
    WHERE NOT (f)-[:PART_OF]->(:Cluster)
    RETURN root, collect(f) AS dislocated_functions
}
WITH *
UNWIND dislocated_functions AS f
CALL (f, root) {
    MATCH (sf:Function)<-[:SUBFUNCTION_OF*]-(f)
    MATCH (c:Cluster)<-[:PART_OF]-(sf)
    MATCH p = (c)-[:PART_OF*]->(root)
    RETURN c, p, length(p) AS length
    ORDER BY length(p) DESC
    LIMIT 1
}
CREATE (f)-[:PART_OF]->(c)''')
		# Position unpositioned direct observables and test results
		db.cypher_query('''CALL () {
    MATCH (root:Cluster)
    WHERE NOT (root)-[:PART_OF]->(:Cluster)
    MATCH (n:DiagnosticTestResult|DirectObservable)
    WHERE NOT (n)-[:PART_OF]->(:Cluster)
    RETURN root, collect(n) AS dislocated_nodes
}
WITH *
UNWIND dislocated_nodes AS n
CALL (n, root) {
    MATCH (f:Function|Hardware)-[:INDICATED_BY|OBSERVED_BY*]->(n)
    WITH collect(f) AS fs
    MATCH (c:Cluster)
    WHERE ALL (f IN fs WHERE (f)-[:PART_OF*]->(c))
    MATCH p = (c)-[:PART_OF*]->(root)
    RETURN c, p, length(p) AS length
    ORDER BY length(p) DESC
    LIMIT 1
}
CREATE (n)-[:PART_OF]->(c)''')
		# Position unpositioned diagnostic tests
		db.cypher_query('''CALL () {
    MATCH (root:Cluster)
    WHERE NOT (root)-[:PART_OF]->(:Cluster)
    MATCH (t:DiagnosticTest)
    WHERE NOT (t)-[:PART_OF]->(:Cluster)
    RETURN root, collect(t) AS dislocated_tests
}
WITH *
UNWIND dislocated_tests AS t
CALL (t, root) {
    MATCH (r:DiagnosticTestResult)<-[:RESULTS_IN*]-(t)
    WITH collect(r) AS rs
    MATCH (c:Cluster)
    WHERE ALL (r IN rs WHERE (r)-[:PART_OF*]->(c))
    MATCH p = (c)-[:PART_OF*]->(root)
    RETURN c, p, length(p) AS length
    ORDER BY length(p) DESC
    LIMIT 1
}
CREATE (t)-[:PART_OF]->(c)''')
		# Position all remaining non-root-cluster nodes in root.
		db.cypher_query('''MATCH (root:Cluster)
WHERE NOT (root)-[:PART_OF]->(:Cluster)
MATCH (n)
WHERE NOT (n)-[:PART_OF]->(:Cluster) AND n <> root
CREATE (n)-[:PART_OF]->(root)''')

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
		func_chains = self._architecture.all_functional_chains
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
			end_node = Function.nodes.get(uid=end_uid)
			opm.set_net(end_node.get_net())

			# add operating mode to required for relations
			for (start_uid, start_chains) in start_nodes:
				start_node = Function.nodes.get(uid=start_uid)
				rel = start_node.required_for.relationship(end_node)
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
