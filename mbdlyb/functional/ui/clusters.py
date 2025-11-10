# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import html
import os
import re
import tempfile
import shutil
from tempfile import TemporaryFile
from pathlib import Path
from typing import Optional, Callable, Union, Type
from nicegui import app, ui, APIRouter, events, run
from neomodel import db
from multiprocessing import Manager, Queue

import mbdlyb.functional as fn
from mbdlyb.functional.gdb import (MBDElement, FunctionalNode, Cluster, Function, DirectObservable, DiagnosticTest,
								   DiagnosticTestResult, Hardware, ObservedError, update_fqn, update_name,
								   RequiredForRelation)
from mbdlyb.ui.helpers import goto, get_object_or_404

from .base import Button, page, confirm, confirm_delete, TableColumn, TableMultiColumn, Table
from .helpers import save_object, save_new_object
from .validation import base_name_validation

router = APIRouter(prefix='/cluster')


# HELPERS METHODS

def download_graph(cluster: Cluster):
	f_cluster: fn.Cluster = Cluster.load(cluster.uid)
	yed_graph = f_cluster.to_yed()
	xml = yed_graph.get_graph()
	ui.download(xml.encode(), f'{cluster.name}.graphml')


def download_bayesnet(cluster: Cluster):
	f_cluster: fn.Cluster = Cluster.load(cluster.uid)
	with TemporaryFile(suffix='.dne') as temp_file:
		f_cluster.save_bn(Path(temp_file.name), overwrite=True)
		ui.download(Path(temp_file.name), f'{cluster.name}.dne')


def copy_cluster(obj) -> None:
	def _copy():
		if obj.at_root:
			query = f"""
				MATCH p = (c: Cluster {{uid: '{obj.uid}'}})<-[:PART_OF*]-(m)-[r]-(l)
				WHERE (l)-[:PART_OF*]->(c) AND type(r) <> 'PART_OF'
				WITH c, collect(p) as paths
				CALL apoc.refactor.cloneSubgraphFromPaths(paths) YIELD output
				WITH c, collect(DISTINCT output) AS outputs

				FOREACH (n IN outputs | SET n.uid = randomUUID())
				WITH c, outputs

				MATCH (copy: Cluster)
				WHERE copy.fqn = c.fqn AND copy IN outputs
				RETURN DISTINCT copy
				"""
		else:
			query = f"""
				MATCH p = (root)<-[:PART_OF]-(c: Cluster {{uid: '{obj.uid}'}})<-[:PART_OF*]-(m)-[r]-(l)
				WHERE (l)-[:PART_OF*]->(c) AND type(r) <> 'PART_OF'
				WITH root, c, collect(p) as paths
				CALL apoc.refactor.cloneSubgraphFromPaths(paths, {{standinNodes: [[root, root]]}}) YIELD output
				WITH c, collect(DISTINCT output) AS outputs

				FOREACH (n IN outputs | SET n.uid = randomUUID())
				WITH c, outputs

				MATCH (copy: Cluster)
				WHERE copy.fqn = c.fqn AND copy IN outputs
				RETURN DISTINCT copy
				"""
		copy, _ = db.cypher_query(query, resolve_objects=True)
		cluster = copy[0][0]

		update_name(cluster)
		goto(f'/cluster/{cluster.uid}/')

	confirm('Copy Cluster',
		 f'Do you want to copy {type(obj).__name__} {obj}?',
		 'Yes', _copy, 'negative')


def move_cluster(obj) -> None:
	def _move(obj, trgt_fqn):
		trgt = get_object_or_404(Cluster, fqn=trgt_fqn)

		query = f"""
			MATCH (m: Cluster {{uid: '{obj.uid}'}})
			OPTIONAL MATCH (m)-[r:PART_OF]->(t:Cluster)
			DELETE r
			WITH m

			MATCH (t:Cluster {{uid: '{trgt.uid}'}})
			CREATE (m)-[:PART_OF]->(t)
			WITH m, t

			OPTIONAL MATCH (m)<-[:PART_OF]-(f:Function)-[r:SUBFUNCTION_OF]->(f2:Function)-[:PART_OF]->(o: Cluster)
			WHERE o.uid <> m.uid
			DELETE r
			RETURN m
			"""
		db.cypher_query(query)

		update_name(obj)
		goto(f'/cluster/{obj.uid}/')

	def _on_cancel():
		dialog.close()
		dialog.clear()

	def _on_confirm(obj, trgt_fqn):
		_move(obj, trgt_fqn)
		_on_cancel()

	query = f"""
		MATCH (c:Cluster)-[:PART_OF*1..]->(root:Cluster)
		MATCH (move:Cluster {{uid: '{obj.uid}'}})-[:PART_OF*1..]->(root)
		WHERE NOT (c)-[:PART_OF*]->(move)
		AND NOT (root)-->()
		AND c.uid <> move.uid
		RETURN DISTINCT c.fqn, root.fqn
		"""
	clusters, _ = db.cypher_query(query, resolve_objects=True)
	clusters = sorted([c[0] for c in clusters] + [clusters[0][1]])

	with ui.dialog(value=True) as dialog, ui.card():
		ui.label('Move Cluster').classes('text-h6')
		ui.label(f'Where do you want to move {type(obj).__name__} {obj} to?',)
		select = ui.select(clusters, with_input=True, label="Cluster").classes('w-60')
		with ui.row():
			ui.button('Cancel', color='secondary', on_click=_on_cancel).props('outline')
			ui.button('Move', color='negative', on_click=lambda: _on_confirm(obj, select.value))
		dialog.on('close', dialog.clear)


def create_mermaid_diagram(cluster: Cluster) -> str:
	class_map = {
		Function: 'Function',
		Hardware: 'Hardware',
		DirectObservable: 'DirectObservable',
		ObservedError: 'ObservedError',
		DiagnosticTest: 'DiagnosticTest',
		DiagnosticTestResult: 'DiagnosticTestResult',
	}
	class_url_map = {
		Cluster: '/cluster/{}/',
		Function: '/function/{}/',
		Hardware: '/hardware/{}/',
		DirectObservable: '/observable/{}/',
		DiagnosticTest: '/test/{}/',
		DiagnosticTestResult: '/test/{}/'
	}

	def m_id(n: MBDElement | str) -> str:
		""" Produce a suitable Mermaid ID, which contains only word characters """
		return re.sub("\\W", "_", n if isinstance(n, str) else n.fqn)

	def m_name(n: MBDElement | str, quote: bool = True) -> str:
		""" Produce a suitable Mermaid name,
		    surrounded by double quotes and where special characters are encoded properly
		"""
		name = html.escape(n if isinstance(n, str) else n.name)
		if quote:
			return f'"{name}"'
		else:
			return name

	def m_url(node: MBDElement, at_root: bool) -> Optional[str]:
		if isinstance(node, FunctionalNode) and not at_root:
			return f'/cluster/{node.get_net().uid}/'
		url = class_url_map.get(node.__class__, None)
		if url:
			url = url.format((node.test if isinstance(node, DiagnosticTestResult) else node).uid)
		return url

	def add_element(e: Union[Function, Hardware, DirectObservable, ObservedError, DiagnosticTest, DiagnosticTestResult],
					url: str = None) -> str:
		e_id = m_id(e)
		diagram.append(f'{e_id}({m_name(e)})')
		diagram.append(f'class {e_id} {class_map[e.__class__]}')
		if url:
			diagram.append('click {} call emitEvent("mermaid_click", "{}")'.format(e_id, url))
		return e_id

	def add_tree(t: dict, at_root: bool = False):
		for name, te in t.items():
			if isinstance(te, dict):
				diagram.append(f'subgraph "{m_id(name)}" ["<b>{name}</b>"]')
				add_tree(te)
				diagram.append('end')
			else:
				url = m_url(te, at_root)
				add_element(te, url)

	def add_to_tree(tree: dict, element: Union[
		Cluster, Function, Hardware, DirectObservable, ObservedError, DiagnosticTest, DiagnosticTestResult]) -> str:
		path_segments = [m_name(s, False) for s in cluster.rfqn(element.fqn).split('.')]
		_tree = tree
		for path_segment in path_segments[:-1]:
			if path_segment not in _tree:
				_tree[path_segment] = dict()
			_tree = _tree[path_segment]
		_tree[element.name] = element
		return m_id(element)

	# start diagram as a flowchart with the main cluster
	diagram = ['flowchart']
	diagram.append(f'subgraph "{cluster.fqn}" ["<b>{cluster.name}</b>"]')
	diagram.append('direction LR')

	# add color mapping
	diagram.append(f'classDef Function fill:{fn.Function.get_color()};')
	diagram.append(f'classDef Hardware fill:{fn.Hardware.get_color()};')
	diagram.append(f'classDef DirectObservable fill:{fn.DirectObservable.get_color()};')
	diagram.append(f'classDef ObservedError fill:{fn.ObservedError.get_color()};')
	diagram.append(f'classDef DiagnosticTest fill:{fn.DiagnosticTest.get_color()};')
	diagram.append(f'classDef DiagnosticTestResult fill:{fn.DiagnosticTestResult.get_color()};')

	tree = dict()

	# add functions and sub-functions, similarly with errors
	fn_dict: dict[str, Function] = dict()
	er_dict: dict[str, ObservedError] = dict()

	relations: dict[Type[fn.FunctionalRelation], set[tuple[str, str]]] = {
		fn.SubfunctionOfRelation: set(),
		fn.RequiredForRelation: set(),
		fn.RealizesRelation: set(),
		fn.ResultsInRelation: set(),
		fn.ObservedByRelation: set(),
		fn.IndicatedByRelation: set(),
		fn.YieldsErrorRelation: set(),
		fn.ReportsErrorRelation: set()
	}

	# collect observables
	for obs in cluster.observables:
		obs_id = add_to_tree(tree, obs)
		for on in obs.get_observed_nodes():
			on_id = add_to_tree(tree, on)
			relations[fn.ObservedByRelation].add((on_id, obs_id))

	# collect tests and test results
	for dt in cluster.tests:
		dt_id = add_to_tree(tree, dt)
		for tr in dt.test_results:
			tr_id = add_to_tree(tree, tr)
			for on in tr.get_observed_nodes():
				if isinstance(on, Function) and on.fqn not in fn_dict:
					fn_dict[on.fqn] = on
				on_id = add_to_tree(tree, on)
				relations[fn.IndicatedByRelation].add((on_id, tr_id))
			relations[fn.ResultsInRelation].add((dt_id, tr_id))

	# collect functions
	for f in cluster.functions:
		f_id = add_to_tree(tree, f)
		fn_dict[f.fqn] = f
		for sf in f.subfunctions:
			sf_id = add_to_tree(tree, sf)
			relations[fn.SubfunctionOfRelation].add((sf_id, f_id))
			fn_dict[sf.fqn] = sf
		for ye in f.yields_error:
			ye_id = add_to_tree(tree, ye)
			relations[fn.YieldsErrorRelation].add((f_id, ye_id))
			if ye not in er_dict:
				er_dict[ye.fqn] = ye
		for er in f.reports_error:
			er_id = add_to_tree(tree, er)
			relations[fn.ReportsErrorRelation].add((f_id, er_id))
			if er not in er_dict:
				er_dict[er.fqn] = er

	# collect hardware
	for hw in cluster.hardware:
		hw_id = add_to_tree(tree, hw)
		for f in hw.realizes:
			relations[fn.RealizesRelation].add((hw_id, m_id(f)))

	# collect required_for relations
	for f in fn_dict.values():
		for dependent_f in f.required_for.all():
			if dependent_f.fqn in fn_dict:
				relations[fn.RequiredForRelation].add((m_id(f), m_id(dependent_f)))

	# draw the schema
	add_tree(tree, True)
	for rt, links in relations.items():
		for x, y in links:
			diagram.append(f'{x} --> |{rt.get_label()}| {y}')

	# close main cluster
	diagram.append('end')
	return '\n'.join(diagram)


# CLUSTER METHODS

class ClusterProps:
	DEFAULT_FUNCTION_NAME = 'MainFunction'

	def __init__(self, add_hw: bool, add_fn: bool):
		self.add_hw = add_hw
		self.hw_name: str = ''
		self.add_fn = add_fn
		self.fn_name: str = ''


def cluster_form(cluster: Cluster, props: ClusterProps, parent: Optional[Cluster] = None, update: bool = True):
	cluster_input = ui.input('Name', validation=base_name_validation(parent, cluster if update else None)) \
		.bind_value(cluster, 'name').props('hide-bottom-space')
	if not update:
		if parent is not None:
			hw_check = ui.checkbox('Add default hardware?').bind_value(props, 'add_hw')
			hw_input = ui.input('Hardware name', validation=base_name_validation(allow_empty=True),
					   placeholder='Leave empty to use cluster name').bind_visibility_from(
						   props, 'add_hw').bind_value(props, 'hw_name').props('hide-bottom-space')
		else:
			hw_input = None
		fn_check = ui.checkbox('Add default function?').bind_value(props, 'add_fn')
		fn_input = ui.input('Function name', validation=base_name_validation(allow_empty=True),
					  placeholder='Leave empty to use \'' + ClusterProps.DEFAULT_FUNCTION_NAME + '\'').bind_visibility_from(
						  props, 'add_fn').bind_value(props, 'fn_name').props('hide-bottom-space')
		if hw_input is not None:
			hw_input.validation['Hardware and Function must have different names'] = \
				lambda s: not props.add_fn or (s != fn_input.value if fn_input.value else s != ClusterProps.DEFAULT_FUNCTION_NAME)
			fn_input.validation['Hardware and Function must have different names'] = \
				lambda s: not props.add_hw or (s != hw_input.value if hw_input.value else s != cluster_input.value)
			cluster_input.on_value_change(fn_input.validate)
			hw_input.on_value_change(fn_input.validate)
			fn_input.on_value_change(hw_input.validate)
			hw_check.on_value_change(fn_input.validate)
			hw_check.on_value_change(hw_input.validate)
			fn_check.on_value_change(fn_input.validate)
			fn_check.on_value_change(hw_input.validate)
		if parent is not None:
			ui.label(
				'If both default hardware and function are added, a realizes relation between them will automatically be created.')


def cluster_create(parent_cluster: Optional[Cluster] = None):
	def _save_cluster(c: Cluster, props: ClusterProps):
		save_object(cluster)
		if parent_cluster is not None:
			c.set_net(parent_cluster)
			update_fqn(c)
		hw = None
		if props.add_hw:
			hw = Hardware(name=props.hw_name if props.hw_name else c.name)
			save_new_object(hw, c, True)
		if props.add_fn:
			fn = Function(name=props.fn_name if props.fn_name else ClusterProps.DEFAULT_FUNCTION_NAME)
			save_new_object(fn, c, True)
			if hw is not None:
				hw.realizes.connect(fn)
		goto(f'/cluster/{c.uid if parent_cluster is None else parent_cluster.uid}/')

	cluster = Cluster(name='')
	props = ClusterProps(False, False)
	title = 'New cluster'
	if parent_cluster is not None:
		title += f' in {parent_cluster.fqn}'
		props.add_hw = True
		props.add_fn = True
	page(title, parent_cluster, [
		Button('Discard', None, 'warning', lambda: goto('/' if parent_cluster is None else f'/cluster/{parent_cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save_cluster(cluster, props))
	])
	cluster_form(cluster, props, parent_cluster, False)


@router.page('/new/')
def cluster_create_root():
	cluster_create()


@router.page('/import/')
def cluster_import_root():
	cluster_import()


@router.page('/{cluster_id}/new/')
def cluster_create_with_parent(cluster_id: str):
	cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	cluster_create(cluster)


@router.page('/{cluster_id}/import/')
def cluster_import_with_parent(cluster_id: str):
	cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	cluster_import(cluster)


@router.page('/{cluster_id}/update/')
def cluster_update(cluster_id: str):
	cluster: Cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	page(f'Update {cluster.fqn}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: save_object(cluster, f'/cluster/{cluster.uid}/'))
	])
	cluster_form(cluster, None if cluster.at_root else cluster.get_net())


@router.page('/{cluster_id}/')
def cluster_details(cluster_id: str):
	cluster: Cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	if cluster.at_root:
		buttons = [[Button(None, 'play_circle', 'positive', lambda: goto(f'/cluster/{cluster.uid}/diagnoser/'), 'Diagnoser'),
				   Button(None, 'design_services', 'positive', lambda: goto(f'/cluster/{cluster.uid}/design_for_diagnostics/'), 'Design for diagnostics')]]
	else:
		buttons = []
	buttons.extend([
			Button(None, 'checklist', 'secondary', lambda: goto(f'/cluster/{cluster.uid}/validator/'), 'Check model'), [
			Button(None, 'account_tree', 'secondary', lambda: download_graph(cluster), 'Download graph'),
			Button(None, 'developer_board', 'secondary', lambda: download_bayesnet(cluster), 'Download Bayesnet')
		],
		Button(None, 'table_view', 'secondary', lambda: goto(f'/cluster/{cluster.uid}/requiredfor/'), 'View table'),
		Button(None, 'upload_file', 'secondary', lambda: goto(f'/cluster/{cluster.uid}/import/'), 'Import cluster'),
		Button(None, 'edit', None, lambda: goto(f'/cluster/{cluster.uid}/update/'), 'Edit cluster'),
		Button(None, 'content_copy', None, lambda: copy_cluster(cluster), 'Copy cluster')])
	if not cluster.at_root:
		buttons.append(Button(None, 'trending_flat', None, lambda: move_cluster(cluster), 'Move cluster'))
	buttons.append(Button(None, 'delete', 'negative',
			   lambda: confirm_delete(cluster, '/' if cluster.at_root else f'/cluster/{cluster.get_net().uid}/'), 'Delete cluster'))
	page(f'Cluster {cluster.name}', cluster, buttons)
	with ui.column().classes('w-full'):
		with ui.card().classes('w-full') as card:
			card.bind_visibility_from(app.storage.general, 'show_diagrams')
			with ui.context_menu():
				save_button = ui.menu_item('Save')
			draw_mermaid_diagram(cluster, save_button)
			ui.on('show_diagram', lambda: draw_mermaid_diagram.refresh(cluster, save_button))
		with ui.card().classes('w-full'):
			Table('Functions', cluster.functions.order_by('name'), [
				TableColumn('Name', 'name', lambda f: f'/function/{f.uid}/', 'fqn'),
				TableMultiColumn('Subfunctions', 'name', lambda f: f'/function/{f.uid}/', 'fqn', 'subfunctions'),
				TableMultiColumn('Required for', 'name', lambda f: f'/function/{f.uid}/', 'fqn', 'required_for'),
				TableMultiColumn('Observed by', 'name', lambda o: f'/observable/{o.uid}/', 'fqn', 'observed_by'),
				TableMultiColumn('Indicated by', 'name', lambda o: f'/test/{o.test.uid}/', 'fqn', 'indicated_by')
			], cluster, [
					  Button(icon='add', color='positive', tooltip='Add function',
							 handler=f'/cluster/{cluster.uid}/function/new/')
				  ], [
					  Button(icon='border_all', color='secondary', tooltip='Modify CPT for {}',
							 handler=lambda f: goto(f'/function/{f.uid}/cpt/')),
					  Button(icon='edit', tooltip='Edit {}', handler=lambda f: goto(f'/function/{f.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete {}',
							 handler=lambda f: confirm_delete(f, f'/cluster/{cluster.uid}/'))
				  ]).show()
		if not cluster.at_root:
			with ui.card().classes('w-full'):
				Table('Hardware', cluster.hardware.order_by('name'), [
					TableColumn('Name', 'name', lambda h: f'/hardware/{h.uid}/', 'fqn'),
					TableColumn('Faults',
								lambda h: ' | '.join([f'{key}: {value}' for key, value in h.fault_rates.items()])),
					TableMultiColumn('Realizes', 'name', lambda f: f'/function/{f.uid}/', 'fqn', 'realizes'),
					TableMultiColumn('Observed by', 'name', lambda o: f'/observable/{o.uid}/', 'fqn', 'observed_by'),
					TableMultiColumn('Indicated by', 'name', lambda o: f'/test/{o.test.uid}/', 'fqn', 'indicated_by'),
				], cluster, [
						  Button(icon='add', color='positive', tooltip='Add hardware',
								 handler=f'/cluster/{cluster.uid}/hardware/new/')
					  ], [
						  Button(icon='edit', tooltip='Edit {}', handler=lambda x: goto(f'/hardware/{x.uid}/update/')),
						  Button(icon='delete', color='negative', tooltip='Delete {}',
								 handler=lambda x: confirm_delete(x, f'/cluster/{cluster.uid}/'))
					  ]).show()
		with ui.card().classes('w-full'):
			Table('Clusters', cluster.subnets.order_by('name'), [
				TableColumn('Name', 'name', lambda c: f'/cluster/{c.uid}/', 'fqn'),
				TableMultiColumn('Functions', 'name', lambda f: f'/function/{f.uid}/', 'fqn', 'functions'),
				TableMultiColumn('Hardware', 'name', lambda h: f'/hardware/{h.uid}/', 'fqn', 'hardware'),
				TableMultiColumn('Observables', 'name', lambda o: f'/observable/{o.uid}/', 'fqn', 'observables'),
				TableMultiColumn('Diagnostic tests', 'name', lambda t: f'/test/{t.uid}/', 'fqn', 'tests')
			], cluster, [
					  Button(icon='add', color='positive', tooltip='Add cluster',
							 handler=f'/cluster/{cluster.uid}/new/')
				  ]).show()
		with ui.card().classes('w-full'):
			Table('Observables', cluster.observables.order_by('name'), [
				TableColumn('Name', 'name', lambda x: f'/observable/{x.uid}/', tooltip='fqn'),
				TableColumn('FPR', 'fp_rate'),
				TableColumn('FNR', 'fn_rate'),
				TableMultiColumn('Observed functions', 'name', lambda x: f'/function/{x.uid}/', 'fqn',
								 'observed_functions'),
				TableMultiColumn('Observed hardware', 'name', lambda x: f'/hardware/{x.uid}/', 'fqn',
								 'observed_hardware')
			], cluster, [
					  Button(icon='add', color='positive', tooltip='Add observable',
							 handler=f'/cluster/{cluster.uid}/observable/new/')
				  ], [
					  Button(icon='border_all', color='secondary', tooltip='Update CPT of {}',
							 handler=lambda x: goto(f'/observable/{x.uid}/cpt/')),
					  Button(icon='edit', tooltip='Edit {}', handler=lambda x: goto(f'/observable/{x.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete {}',
							 handler=lambda x: confirm_delete(x, f'/cluster/{cluster.uid}/'))
				  ]).show()
		with ui.card().classes('w-full'):
			Table('Diagnostic tests', cluster.tests.order_by('name'), [
				TableColumn('Name', 'name', lambda t: f'/test/{t.uid}/', 'fqn'),
				TableColumn('Costs', lambda t: f'{sum(t.fixed_cost.values())} ({", ".join(t.fixed_cost.keys())})'),
				TableMultiColumn('Results', lambda r: f'{r.name} ({", ".join(o.name for o in r.get_observed_nodes())})',
								 list_attribute='test_results')
			], cluster, [
					  Button(icon='add', color='positive', tooltip='Add diagnostic test',
							 handler=f'/cluster/{cluster.uid}/test/new/')
				  ], [
					  Button(icon='edit', tooltip='Edit {}', handler=lambda t: goto(f'/test/{t.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete {}',
							 handler=lambda t: confirm_delete(t, f'/cluster/{cluster.uid}/'))
				  ]).show()
		with ui.card().classes('w-full'):
			Table('Operating mode selectors', cluster.operating_modes.order_by('name'), [
				TableColumn('Name', 'name', lambda x: f'/opm/{x.uid}/', 'fqn'),
				TableMultiColumn('Modes', lambda x: x, list_attribute='operating_modes')
			], cluster, [
					  Button(icon='add', color='positive', tooltip='Add operating mode',
							 handler=f'/cluster/{cluster.uid}/opm/new/')
				  ], [
					  Button(icon='edit', tooltip='Edit {}', handler=lambda x: goto(f'/opm/{x.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete {}',
							 handler=lambda x: confirm_delete(x, f'/cluster/{cluster.uid}/'))
				  ]).show()

@ui.refreshable
def draw_mermaid_diagram(cluster: Cluster, save_button: ui.menu_item):
	mermaid_diagram = create_mermaid_diagram(cluster)
	mermaid_config = {
		'securityLevel': 'loose',
		'theme': 'neutral',
		'themeVariables': {
			'fontSize': '14px'
		},
		'layout': 'elk',
		'elk': {
			'nodePlacementStrategy': 'LINEAR_SEGMENTS'
		},
		'maxTextSize': 1e6
	}
	mm = ui.mermaid(mermaid_diagram, config=mermaid_config).classes('w-full')
	ui.on('mermaid_click', lambda event: goto(event.args))

	def export_start():
		ui.run_javascript(f'''
			svg = document.querySelector("#c" + {mm.id} + " > svg");
			emitEvent('mermaid_export', svg.outerHTML);
		''')
	save_button.on_click(export_start)

	def export_store_svg(svg):
		ui.download(svg.encode(), f'{cluster.name}.svg')
	ui.on('mermaid_export', lambda evt: export_store_svg(evt.args))


# REQUIRED_FOR RELATION TABLE

@router.page('/{cluster_id}/requiredfor/')
def required_for_table(cluster_id: str):
	def _create_new_rf_rel(x: Function, y: Function):
		rel = RequiredForRelation(x, y)
		rel.save(f'/cluster/{cluster_id}/requiredfor/')

	cluster: Cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	functions, _ = db.cypher_query(f'''MATCH
	(f:Function)-[:PART_OF]->(:Cluster)(()-[:PART_OF]->()){{1,1}}(:Cluster {{uid: '{cluster.uid}'}})
RETURN f
ORDER BY f.fqn''', resolve_objects=True)
	page(f'Required for relations of cluster {cluster.fqn}', cluster, [
		Button('Close', None, None, lambda: goto(f'/cluster/{cluster.uid}/'))])
	with ui.grid(columns=len(functions) + 1).classes('gap-0'):
		ui.label()
		for f in functions:
			ui.link(f[0].name, f'/function/{f[0].uid}/').classes('border p-1 font-bold').tooltip(f[0].fqn)
		for f in functions:
			ui.link(f[0].name, f'/function/{f[0].uid}/').classes('border p-1 font-bold').tooltip(f[0].fqn)
			for g in functions:
				eligable_for_relation = f[0] != g[0]
				needs_repair = f[0].has_subfunctions or g[0].has_subfunctions
				relation = f[0].required_for.relationship(g[0])
				disabled = False
				color = None
				classes = ''
				props = ''
				on_click: Optional[Callable] = None
				if eligable_for_relation and relation is not None:
					if needs_repair:
						icon, color = 'construction', 'warning'
						on_click = lambda x=f[0], y=g[0]: goto(f'/function/{x.uid}/requiredfor/{y.uid}/repair/')
					else:
						icon, color = 'edit', 'positive'
						on_click = lambda x=f[0], y=g[0]: goto(f'/function/{x.uid}/requiredfor/{y.uid}/update/')
				elif eligable_for_relation and not needs_repair:
					icon, classes, props = 'add_circle', 'flat', 'text-color=primary'
					on_click = lambda x=f[0], y=g[0]: _create_new_rf_rel(x, y)
				else:
					icon, disabled = 'do_not_disturb_on', True
				btn = ui.button(icon=icon, color=color).tooltip(f'{f[0].fqn} -> {g[0].fqn}')
				if classes:
					btn.classes(classes)
				if props:
					btn.props(props)
				if disabled:
					btn.disable()
				elif on_click is not None:
					btn.on_click(on_click)


# CLUSTER IMPORT

class ClusterImportProps:
	def __init__(self, main_cluster_name: str, parent: Cluster = None, capella_architecture: str = None,
				 default_fn_tests: bool = False, default_hl_fn_tests: bool = False, default_hw_tests: bool = False,
				 default_ll_hw_tests: bool = False, import_opms: bool = False):
		self.main_cluster_name = main_cluster_name
		self.parent = parent
		self.capella_architecture = capella_architecture
		self.default_fn_tests = default_fn_tests
		self.default_hw_tests = default_hw_tests
		self.default_hl_fn_tests = default_hl_fn_tests
		self.default_ll_hw_tests = default_ll_hw_tests
		self.import_opms = import_opms
		self.files: list[Path] = []


# async def handle_upload(e: events.MultiUploadEventArguments, props: ClusterImportProps, suffix: str):
# 	if props.parent is not None:
# 		if props.parent.has_node(props.main_cluster_name):
# 			ui.notify('Name already exists!')
# 			return
# 	else:
# 		if props.main_cluster_name in {c.name for c in Cluster.nodes.has(net=False)}:
# 			ui.notify('Name already exists!')
# 			return
# 	for f in e.files:
# 		with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
# 			tf.write(await f.read())
# 			props.files.append(Path(tf.name))


async def handle_capella_upload(e: events.MultiUploadEventArguments, props: ClusterImportProps):
	if props.parent is not None:
		if props.parent.has_node(props.main_cluster_name):
			ui.notify('Name already exists!')
			return
	else:
		if props.main_cluster_name in {c.name for c in Cluster.nodes.has(net=False)}:
			ui.notify('Name already exists!')
			return

	for f in e.files:
		fd, tmp_path = tempfile.mkstemp(suffix=Path(f.name).suffix)
		os.close(fd)

		new_tmp_path = Path(tmp_path).with_name(f.name)
		shutil.move(tmp_path, new_tmp_path)

		with new_tmp_path.open('wb') as new_tf:
			new_tf.write(await f.read())

		props.files.append(new_tmp_path)


def cluster_import_form(props: ClusterImportProps):
	# ui.input('Main cluster name', validation=base_name_validation(props.parent, None)) \
	# 	.bind_value(props,'main_cluster_name').props('hide-bottom-space')
	# ui.upload(label='Dependency table', multiple=True, on_multi_upload=lambda e: handle_upload(e, props, '.xlsx'),
	# 		  auto_upload=True).props('accept=.xlsx')
	# ui.upload(label='Connections', multiple=True, on_multi_upload=lambda e: handle_upload(e, props, '.json'),
	# 		  auto_upload=True).props('accept=.json')

	# ui.label("Capella import").style('font-size: 140%')
	ui.upload(label='Capella', multiple=True, on_multi_upload=lambda e: handle_capella_upload(e, props), auto_upload=True)
	ui.select(["logical", "physical"], label="Capella architecture").bind_value(props, 'capella_architecture').classes('w-60')
	with ui.row():
		ui.checkbox(text="Default tests on functions").bind_value(props, "default_fn_tests")
		ui.checkbox(text="Default tests on high-level functions (functions with subfunctions)").bind_value(props,
																										   "default_hl_fn_tests")
	with ui.row():
		ui.checkbox(text="Default tests on hardware").bind_value(props, "default_hw_tests")
		ui.checkbox(
			text="Default tests on low-level hardware (hardware realizing functions without subfunctions)").bind_value(
			props, "default_ll_hw_tests")
	with ui.row():
		ui.checkbox(text="Import operating modes").bind_value(props, "import_opms")


def cluster_import(parent_cluster: Optional[Cluster] = None):
	async def _save(props: ClusterImportProps, queue: Queue, progress_bar: ui.linear_progress, timer: ui.timer):
		ui.notify('Importing model, please wait. This could take a long time, depending on the size of your model.')
		if props.capella_architecture:
			progress_bar.visible = True
			timer.active = True
			c: Cluster = await run.io_bound(
				Cluster.load_capella,
				props.capella_architecture,
				props.default_fn_tests,
				props.default_hl_fn_tests,
				props.default_hw_tests,
				props.default_ll_hw_tests,
				props.import_opms,
				*props.files,
				queue=queue)
			timer.active = False
			progress_bar.visible = False
		else:
			await run.io_bound(Cluster.load_xls, props.main_cluster_name, *props.files)
			c: Cluster = Cluster.nodes.has(net=False).get(name=props.main_cluster_name)
		if props.parent is not None:
			c.net.connect(props.parent)
		for f in props.files:
			f.unlink()
		goto(f'/cluster/{c.uid}/')

	props = ClusterImportProps('', parent_cluster)

	queue: Queue = Manager().Queue()
	progress_bar = ui.linear_progress(value=0., show_value=False, size='10px').classes('rounded').props(
		'animation-speed:1000')
	progress_bar.visible = False
	timer = ui.timer(0.1,
					 callback=lambda: progress_bar.set_value(queue.get() if not queue.empty() else progress_bar.value),
					 active=False)

	title = 'Capella import'
	if parent_cluster is not None:
		title += f' into {parent_cluster.fqn}'
	page(title, parent_cluster, [
		Button('Discard', None, 'warning', lambda: goto('/' if parent_cluster is None else f'/cluster/{parent_cluster.uid}/')),
		Button('Save', None, 'primary', lambda: _save(props, queue, progress_bar, timer))
	])
	cluster_import_form(props)
