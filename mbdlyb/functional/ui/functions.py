# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""

from neomodel import db
from nicegui import ui, APIRouter

import mbdlyb
from mbdlyb.functional.gdb import Cluster, Function, DirectObservable, DiagnosticTest
from mbdlyb.ui.helpers import get_object_or_404, get_relation_or_404

from .base import Button, page, build_table, confirm_delete, confirm_delete_relation
from .helpers import (goto, save_object, save_new_object, SubfunctionOfRelation, RequiredForRelation, ObservesRelation,
					  TestsRelation)
from .validation import base_name_validation


router = APIRouter()


def function_form(function: Function, cluster: Cluster, update: bool = True):
	ui.input('Name', validation=base_name_validation(cluster, function if update else None)) \
		.bind_value(function, 'name').props('hide-bottom-space')


@router.page('/cluster/{cluster_id}/function/new/')
def function_create(cluster_id: str):
	cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return

	function = Function(name='')
	page(f'New function in {cluster.name}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: save_new_object(function, cluster))
	])
	function_form(function, cluster, False)


@router.page('/function/{function_id}/update/')
def function_update(function_id: str):
	function: Function = get_object_or_404(Function, uid=function_id)
	if function is None:
		return
	cluster = function.get_net()
	page(f'Update {function.fqn}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: save_object(function, f'/cluster/{cluster.uid}/'))
	])
	function_form(function, cluster)


@router.page('/function/{function_id}/')
def function_details(function_id: str):
	function: Function = get_object_or_404(Function, uid=function_id)
	if function is None:
		return
	cluster: Cluster = function.get_net()
	buttons = [
		Button(None, 'edit', None, lambda: goto(f'/function/{function.uid}/update/'), 'Edit function'),
		Button(None, 'delete', 'negative', lambda: confirm_delete(function, f'/cluster/{cluster.uid}/'),
			   'Delete function')]
	page(f'Function {function.name}', cluster, buttons)
	for label, rel, target, fn in (('Subfunction of', 'subfunction_of', 'function', None),
								   ('Realized by', 'realized_by', 'hardware', None),
								   ('Requires', 'requires', 'function', None),
								   ('Indicated by', 'indicated_by', 'test', lambda x: x.results_from.single())):
		if function.__getattribute__(rel):
			with ui.row():
				ui.label(label)
				for sf in function.__getattribute__(rel).order_by('name'):
					ui.link(sf.name, f'/{target}/{(sf if fn is None else fn(sf)).uid}/').tooltip(sf.fqn)
	with ui.grid(columns='1fr 1fr').classes('w-full'):
		with ui.card():
			build_table('Subfunctions', [
				('FQN', 'fqn'), ('Weight', ('subfunction_of', lambda sf: sf.relationship(function).weight))
			], function.subfunctions.order_by('fqn'),
						detail_url='/function/{}/',
						create_url=f'/function/{function.uid}/subfunction/new/',
						edit_fn=lambda x: goto(f'/function/{function.uid}/subfunction/{x.uid}/update/'),
						delete_fn=lambda x: confirm_delete_relation('subfunctions', function, x,
																	f'/function/{function.uid}/'))
		if not function.has_subfunctions or function.required_for:
			with ui.card():
				build_table('Required for', [
					('FQN', 'fqn'), ('Weight', ('requires', lambda df: df.relationship(function).weight))
				], function.required_for.order_by('fqn'),
							detail_url='/function/{}/',
							create_url=f'/function/{function.uid}/requiredfor/new/' if not function.has_subfunctions else None,
							edit_fn=lambda x: goto(f'/function/{function.uid}/requiredfor/{x.uid}/update/'),
							delete_fn=lambda x: confirm_delete_relation('required_for', function, x,
																		f'/function/{function.uid}/'))
		with ui.card():
			build_table('Observed by', [
				('FQN', 'fqn'), ('Weight', ('observed_functions', lambda sf: sf.relationship(function).weight))
			], function.observed_by.order_by('fqn'),
						detail_url='/observable/{}/',
						create_url=f'/function/{function.uid}/observedby/new/',
						edit_fn=lambda x: goto(f'/function/{function.uid}/observedby/{x.uid}/update/'),
						delete_fn=lambda x: confirm_delete_relation('observed_by', function, x,
																	f'/function/{function.uid}/'))


# SUBFUNCTION RELATION

def subfunction_relation_form(sf_relation: SubfunctionOfRelation):
	options, _ = db.cypher_query(
		f'''MATCH
    (f:Function {{uid: '{sf_relation.start_uid}'}})-[:PART_OF]->(c:Cluster),
    (g:Function)-[r:PART_OF]->{{0,99}}(c)
WHERE
    NOT (g)-[:SUBFUNCTION_OF]->(f) AND g <> f
 RETURN g
 ORDER BY size(r), g.fqn''', resolve_objects=True)
	options_dict = dict() if sf_relation.end_uid is None else {sf_relation.end_uid: sf_relation.get_end_node().name}
	ui.select({**options_dict, **{sf[0].uid: sf[0].fqn for sf in options}}, label='Subfunction',
			  with_input=True).bind_value(sf_relation, 'end_uid')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(sf_relation, 'weight')


@router.page('/function/{function_id}/subfunction/new/')
def subfunction_relation_create(function_id: str):
	function = get_object_or_404(Function, uid=function_id)
	if function is None:
		return

	sf_relation = SubfunctionOfRelation(function, node_type=Function)
	page(f'Add subfunction to {function}', function.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: sf_relation.save(f'/function/{function.uid}/'))
	])
	subfunction_relation_form(sf_relation)


@router.page('/function/{function_id}/subfunction/{subfunction_id}/update/')
def subfunction_relation_update(function_id: str, subfunction_id: str):
	function = get_object_or_404(Function, uid=function_id)
	subfunction = get_object_or_404(Function, uid=subfunction_id)
	relation = get_relation_or_404(function, subfunction, 'subfunctions')
	if None in (function, subfunction, relation):
		return

	sf_relation = SubfunctionOfRelation(function, subfunction, weight=relation.weight)
	page(f'Update {relation}', function.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: sf_relation.save(f'/function/{function.uid}/'))
	])
	subfunction_relation_form(sf_relation)


@router.page('/function/{function_id}/subfunction/')
def subfunction_relation_table(function_id: str):
	def _save(function: Function, sfs: dict[str, dict]):
		function.subfunctions.disconnect_all()
		for sf in sfs.values():
			if sf['checked']:
				SubfunctionOfRelation(function, sf['fn'], weight=sf['weight']).save()
		goto(f'/cluster/{function.get_net().uid}/')

	function: Function = get_object_or_404(Function, uid=function_id)
	if function is None:
		return
	cluster: Cluster = function.get_net()
	eligable_sfs = {
		fn.uid: {'fn': fn, 'checked': False, 'weight': 1.0} for sc in cluster.subnets.order_by('name') for fn in
		sc.functions.order_by('name')
	}
	functions, _ = db.cypher_query(
		f'''MATCH
	(f:Function {{uid: '{function.uid}'}})-[:PART_OF]->(c:Cluster),
	(sc:Cluster)-[r:PART_OF]->{{0,99}}(c)
RETURN sc.fqn AS cluster, COLLECT {{
	MATCH (g:Function)-[:PART_OF]->(sc)
	WHERE g <> f
	RETURN g
	ORDER BY g.fqn
}} AS functions
ORDER BY size(r), sc.fqn''', resolve_objects=True)
	groups: dict[str, list[str]] = dict()
	eligable_sfs: dict[str, dict] = dict()
	for r in functions:
		if r[0] not in groups:
			groups[r[0]] = []
		for fn in r[1][0]:
			groups[r[0]].append(fn.uid)
			eligable_sfs[fn.uid] = {'fn': fn, 'checked': False, 'weight': 1.0}
	for sf in function.subfunctions.all():
		sf_r = function.subfunctions.relationship(sf)
		e = eligable_sfs[sf.uid]
		e['checked'] = True
		e['weight'] = sf_r.weight
	page(f'Subfunctions of {function}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: _save(function, eligable_sfs))
	])
	with ui.grid(columns=2).classes('gap-0'):
		for c, sfs in groups.items():
			if not sfs:
				continue
			ui.label(c).classes('col-span-full text-h6 border p-2')
			for f_uid in sfs:
				fn = eligable_sfs[f_uid]
				cb = ui.checkbox(fn['fn'].fqn).bind_value(fn, 'checked').props('size=xs')
				ui.slider(min=.0, max=1., step=.01).bind_enabled_from(cb, 'value').bind_value(fn, 'weight').props(
					'label-always').classes('p-2')


# REQUIRED_FOR RELATION

def requiredfor_relation_form(rf_relation: RequiredForRelation, root: Cluster):
	options, _ = db.cypher_query(
		f'''MATCH
    (f:Function {{uid: '{rf_relation.start_uid}'}}),
    (g:Function WHERE g.uid <> '{rf_relation.start_uid}')-[:PART_OF*]->(:Cluster {{uid: '{root.uid}'}})
WHERE
    NOT (g)<-[:SUBFUNCTION_OF]-(:Function) AND NOT (f)-[:REQUIRED_FOR]->(g)
 RETURN g''', resolve_objects=True)
	options_dict = dict() if rf_relation.end_uid is None else {rf_relation.end_uid: rf_relation.get_end_node().fqn}
	ui.select({**options_dict, **{sf[0].uid: sf[0].fqn for sf in options}}, label='Dependent function',
			  with_input=True).bind_value(rf_relation, 'end_uid')
	opm_values = _get_defined_opms(rf_relation.get_start_node(), rf_relation.get_end_node())
	ui.select(opm_values, label='Operating mode(s)', with_input=True, multiple=True).bind_value(rf_relation,
																								'operating_modes').props(
		'use-chips')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(rf_relation, 'weight')


@router.page('/function/{function_id}/requiredfor/new/')
def requiredfor_relation_create(function_id: str):
	function = get_object_or_404(Function, uid=function_id)
	if function is None:
		return

	rf_relation = RequiredForRelation(function, node_type=Function)
	page(f'Add required_for to {function}', function.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: rf_relation.save(f'/function/{function.uid}/'))
	])
	root = function.get_root()
	requiredfor_relation_form(rf_relation, root)


@router.page('/function/{function_id}/requiredfor/{depfunction_id}/update/')
def requiredfor_relation_update(function_id: str, depfunction_id: str):
	function = get_object_or_404(Function, uid=function_id)
	depfunction = get_object_or_404(Function, uid=depfunction_id)
	relation: RequiredForRelation = get_relation_or_404(function, depfunction, 'required_for')
	if None in (function, depfunction, relation):
		return

	rf_relation = RequiredForRelation(function, depfunction, weight=relation.weight,
									  operating_modes=relation.operating_modes)
	page(f'Update {relation}', function.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: rf_relation.save(f'/function/{function.uid}/'))
	])
	root = function.get_root()
	requiredfor_relation_form(rf_relation, root)


def _get_defined_opms(f1: Function, f2: Function = None) -> list[str]:
	uids = [f1.uid]
	if f2 is not None:
		uids.append(f2.uid)
	result, _ = db.cypher_query(
		f'MATCH (n WHERE n.uid IN {uids})-[:PART_OF]->(:Cluster)(()-[:PART_OF]->()){{0,99}}(c:Cluster)<-[:PART_OF]-(o:OperatingMode) RETURN DISTINCT o',
		resolve_objects=True)
	return [om for r in result for om in r[0].operating_modes]


# REQUIRED FOR RELATION REPAIR

class FormRequiredForRelation(RequiredForRelation):
	def __init__(self, idx: int, start_node: Function, end_node: Function = None, weight: float = None,
				 operating_modes: list[str] = None):
		self.idx = idx
		super().__init__(start_node, end_node, weight=weight, operating_modes=operating_modes)

	def __eq__(self, other):
		if isinstance(other, FormRequiredForRelation):
			return self.idx == other.idx


def add_rfr(rf_relations: list[FormRequiredForRelation], initial: FormRequiredForRelation):
	rf_relations.append(FormRequiredForRelation(max(r.idx for r in rf_relations) + 1, initial.get_original_start_node(),
												initial.get_original_end_node(), initial.weight,
												initial.operating_modes))
	repair_rfr_table.refresh()


def drop_rfr(rf_relations: list[FormRequiredForRelation], rf_relation: FormRequiredForRelation):
	rf_relations.remove(rf_relation)
	repair_rfr_table.refresh()


@ui.refreshable
def repair_rfr_table(rf_relations: list[FormRequiredForRelation], initial_relation: FormRequiredForRelation,
					 start_options: dict[str, str] = None,
					 end_options: dict[str, str] = None):
	for idx, rfr in enumerate(rf_relations):
		ui.label('Correction').classes('font-bold')
		if start_options:
			ui.select(start_options, label='Required function', with_input=True).bind_value(rfr, 'start_uid')
		else:
			ui.label(initial_relation.get_original_start_node().fqn)
		ui.label(' -> ')
		if end_options:
			ui.select(end_options, label='Dependent function', with_input=True).bind_value(rfr, 'end_uid')
		else:
			ui.label(initial_relation.get_original_end_node().fqn)
		ui.select(initial_relation.operating_modes, label='Operating mode(s)', with_input=True,
				  multiple=True).bind_value(rfr, 'operating_modes').props('use-chips')
		ui.number('Weight', min=0., max=1., step=.01).bind_value(rfr, 'weight')
		if idx > 0:
			ui.icon('delete', color='negative').on('click', lambda r=rfr: drop_rfr(rf_relations, r)).classes(
				'self-end cursor-pointer')
		else:
			ui.label()


def requiredfor_relation_repair_form(rf_relations: list[FormRequiredForRelation],
									 initial_relation: FormRequiredForRelation):
	q: str = '''MATCH (:Function {{uid: \'{}\'}})(()<-[:SUBFUNCTION_OF]-()){{1,99}}(f:Function) 
WHERE NOT (f)<-[:SUBFUNCTION_OF]-(:Function)
RETURN f'''
	start_node: Function = initial_relation.get_start_node()
	end_node: Function = initial_relation.get_end_node()
	start_options = end_options = None
	if start_node.has_subfunctions:
		start_options, _ = db.cypher_query(q.format(initial_relation.start_uid), resolve_objects=True)
		start_options = {f[0].uid: f[0].fqn for f in start_options}
	if end_node.has_subfunctions:
		end_options, _ = db.cypher_query(q.format(initial_relation.end_uid), resolve_objects=True)
		end_options = {f[0].uid: f[0].fqn for f in end_options}
	with ui.grid(columns='auto 1fr auto 1fr auto auto auto').classes('w-full vertical-bottom'):
		ui.label('Invalid').classes('font-bold')
		ui.label(start_node.fqn)
		ui.label(' -> ')
		ui.label(end_node.fqn)
		ui.label(' | '.join(initial_relation.operating_modes))
		ui.label(str(initial_relation.weight))
		ui.icon('add', color='primary').on('click', lambda: add_rfr(rf_relations, initial_relation)).classes(
			'cursor-pointer')
		repair_rfr_table(rf_relations, initial_relation, start_options, end_options)


@router.page('/function/{function_id}/requiredfor/{depfunction_id}/repair/')
def requiredfor_relation_repair(function_id: str, depfunction_id: str):
	def _save(rf_relations: list[FormRequiredForRelation], url: str):
		for r in rf_relations:
			r.save()
		goto(url)

	function: Function = get_object_or_404(Function, uid=function_id)
	depfunction: Function = get_object_or_404(Function, uid=depfunction_id)
	relation: mbdlyb.RequiredForRelation = get_relation_or_404(function, depfunction, 'required_for')
	if None in (function, depfunction, relation):
		return

	rf_relation = FormRequiredForRelation(0, function, depfunction, relation.weight, relation.operating_modes)
	rf_relations = [rf_relation]
	page(f'Repair {relation}', function.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: _save(rf_relations, f'/function/{function.uid}/'))
	])
	requiredfor_relation_repair_form(rf_relations, rf_relation)


# OBSERVED BY RELATION

def observes_relation_form(o_relation: ObservesRelation):
	options, _ = db.cypher_query(
		f'''MATCH
    (f:Function {{uid: '{o_relation.start_uid}'}})-[:PART_OF]->(:Cluster)(()-[:PART_OF]->()){{0,99}}(:Cluster)<-[PART_OF]-(o:DirectObservable)
WHERE
    NOT (f)-[:OBSERVED_BY]->(o)
 RETURN o''', resolve_objects=True)
	options_dict = dict() if o_relation.end_uid is None else {o_relation.end_uid: o_relation.get_end_node().fqn}
	ui.select({**options_dict, **{sf[0].uid: sf[0].fqn for sf in options}}, label='Observer',
			  with_input=True).bind_value(o_relation, 'end_uid')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(o_relation, 'weight')


@router.page('/function/{function_id}/observedby/new/')
def observes_relation_create(function_id: str):
	function = get_object_or_404(Function, uid=function_id)
	if function is None:
		return

	o_relation = ObservesRelation(function, node_type=DirectObservable)
	page(f'Add observed_by to {function}', function.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: o_relation.save(f'/function/{function.uid}/'))
	])
	observes_relation_form(o_relation)


@router.page('/function/{function_id}/observedby/{observable_id}/update/')
def observes_relation_update(function_id: str, observable_id: str):
	function = get_object_or_404(Function, uid=function_id)
	observable = get_object_or_404(DirectObservable, uid=observable_id)
	relation = get_relation_or_404(function, observable, 'observed_by')
	if None in (function, observable, relation):
		return

	o_relation = ObservesRelation(function, observable, weight=relation.weight)
	page(f'Update {relation}', function.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: o_relation.save(f'/function/{function.uid}/'))
	])
	observes_relation_form(o_relation)


# TESTED BY RELATION

def tests_relation_form(t_relation: TestsRelation):
	options, _ = db.cypher_query(
		f'''MATCH
    (f:Function {{uid: '{t_relation.start_uid}'}})-[:PART_OF]->(:Cluster)(()-[:PART_OF]->()){{0,99}}(:Cluster)<-[PART_OF]-(t:DiagnosticTest)
WHERE
    NOT (f)-[:TESTED_BY]->(t)
 RETURN t''', resolve_objects=True)
	options_dict = dict() if t_relation.end_uid is None else {t_relation.end_uid: t_relation.get_end_node().fqn}
	ui.select({**options_dict, **{sf[0].uid: sf[0].fqn for sf in options}}, label='Diagnostic test',
			  with_input=True).bind_value(t_relation, 'end_uid')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(t_relation, 'weight')


@router.page('/function/{function_id}/testedby/new/')
def tests_relation_create(function_id: str):
	function = get_object_or_404(Function, uid=function_id)
	if function is None:
		return

	t_relation = TestsRelation(function, node_type=DiagnosticTest)
	page(f'Add tested_by to {function}', function.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: t_relation.save(f'/function/{function.uid}/'))
	])
	tests_relation_form(t_relation)


@router.page('/function/{function_id}/testedby/{test_id}/update/')
def observes_relation_update(function_id: str, test_id: str):
	function = get_object_or_404(Function, uid=function_id)
	test = get_object_or_404(DiagnosticTest, uid=test_id)
	relation = get_relation_or_404(function, test, 'tested_by')
	if None in (function, test, relation):
		return

	t_relation = TestsRelation(function, test, weight=relation.weight)
	page(f'Update {relation}', function.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/function/{function.uid}/')),
		Button('Save', 'save', None, lambda: t_relation.save(f'/function/{function.uid}/'))
	])
	tests_relation_form(t_relation)
