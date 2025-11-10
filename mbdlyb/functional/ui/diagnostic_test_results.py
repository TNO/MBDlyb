# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import mbdlyb.functional as fn

from nicegui import ui, APIRouter
from neomodel import db

from mbdlyb.functional.gdb import Cluster, DiagnosticTest, DiagnosticTestResult, Function, Hardware
from mbdlyb.ui.helpers import get_object_or_404

from .base import Button, page, goto
from .helpers import save_object, save_new_object, IndicatesRelation, node_cpt, StateTable, state_table
from .validation import base_name_validation

router = APIRouter()


TEST_RESULT_RELATION_MAP = {
	Function: 'indicated_functions',
	Hardware: 'indicated_hardware'
}


def test_result_form(test_result: DiagnosticTestResult, states: StateTable, cluster: Cluster, update: bool = True):
	ui.input('Name', validation=base_name_validation(cluster, test_result if update else None)) \
		.bind_value(test_result, 'name').props('hide-bottom-space')
	ui.number('False positive rate', min=0., max=1., step=.0001).bind_value(test_result, 'fp_rate')
	ui.number('False negative rate', min=0., max=1., step=.0001).bind_value(test_result, 'fn_rate')
	state_table(states, fn.DiagnosticTestResult.DEFAULT_STATES)


@router.page('/test/{test_id}/results/new/')
def test_result_create(test_id: str):
	def _save(test_result: DiagnosticTestResult, states: StateTable, test: DiagnosticTest, cluster: Cluster):
		test_result.states = states.to_list()
		save_new_object(test_result, cluster)
		test_result.results_from.connect(test)
		goto(f'/test/{test.uid}/')

	test: DiagnosticTest = get_object_or_404(DiagnosticTest, uid=test_id)
	cluster = test.get_net()
	if cluster is None:
		return
	test_result = DiagnosticTestResult(name='')
	states = StateTable.from_list(test_result.states_)
	page(f'New diagnostic test result in {test.name}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/test/{test.uid}/')),
		Button('Save', 'save', None, lambda: _save(test_result, states, test, cluster))
	])
	test_result_form(test_result, states, cluster, False)


@router.page('/test_result/{result_id}/update/')
def test_update(result_id: str):
	def _save(test_result: DiagnosticTestResult, states: StateTable, red_url: str):
		test_result.states = states.to_list()
		save_object(test_result, red_url)

	test_result: DiagnosticTestResult = get_object_or_404(DiagnosticTestResult, uid=result_id)
	if test_result is None:
		return
	cluster = test_result.get_net()
	test: DiagnosticTest = test_result.results_from.single()
	states = StateTable.from_list(test.states_)
	page(f'Update diagnostic test result {test_result.fqn}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/test/{test.uid}/')),
		Button('Save', 'save', None, lambda: _save(test_result, states, f'/test/{test.uid}/'))
	])
	test_result_form(test_result, states, cluster)


@router.page('/test_result/{result_id}/tests/')
def testresult_relation_table(result_id: str):
	def _save(result: DiagnosticTestResult, inodes: dict[Cluster, dict[str, dict[str, dict]]]):
		result.indicated_functions.disconnect_all()
		result.indicated_hardware.disconnect_all()
		for node_type_groups in inodes.values():
			for node_type, nodes in node_type_groups.items():
				for n_uid, node in nodes.items():
					if node['checked']:
						IndicatesRelation(node['n'], result, weight=node['weight']).save()
		goto(f'/test/{result.results_from.single().uid}/')

	result: DiagnosticTestResult = get_object_or_404(DiagnosticTestResult, uid=result_id)
	if result is None:
		return
	cluster: Cluster = result.get_net()
	test: DiagnosticTest = result.results_from.single()
	qr, _ = db.cypher_query(f'''MATCH
	(:DiagnosticTestResult {{uid: '{result.uid}'}})-[:PART_OF]->(:Cluster)<-[:PART_OF*0..]-(c:Cluster)
RETURN c,
	COLLECT {{ MATCH (c)<-[:PART_OF]-(f:Function) RETURN f ORDER BY f.fqn }} as functions,
	COLLECT {{ MATCH (c)<-[:PART_OF]-(h:Hardware) RETURN h ORDER BY h.fqn }} as hardware
ORDER BY c.fqn''', resolve_objects=True)
	indicates = {
		r[0]: {
			ntype: {
				n.uid: {
					'n': n, 'checked': False, 'weight': 1.0
				} for n in r[elem][0]
			} for ntype, elem in [('Function', 1), ('Hardware', 2)]
		} for r in qr
	}
	for tn in result.get_observed_nodes():
		tn_r = result.__getattribute__(TEST_RESULT_RELATION_MAP[type(tn)]).relationship(tn)
		e = indicates[tn.get_net()][tn.__class__.__name__][tn.uid]
		e['checked'] = True
		e['weight'] = tn_r.weight
	page(f'Tested items of {result}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/test/{test.uid}/')),
		Button('Save', 'save', None, lambda: _save(result, indicates))
	])
	with ui.grid(columns=2).classes('gap-0'):
		for c, node_types in indicates.items():
			ui.label(c.fqn).classes('col-span-full text-h6 border p-2')
			for node_type, nodes in node_types.items():
				ui.label(node_type).classes('col-span-full text-weight-bold border p-1')
				for n_uid, n in nodes.items():
					cb = ui.checkbox(n['n'].fqn).bind_value(n, 'checked').props('size=xs')
					ui.slider(min=.0, max=1., step=.01).bind_enabled_from(cb, 'value').bind_value(n, 'weight').props(
						'label-always').classes('p-2')


@router.page('/test_result/{test_id}/cpt/')
def test_cpt(test_id: str):
	test_result = get_object_or_404(DiagnosticTestResult, uid=test_id)
	if test_result is None:
		return
	node_cpt(DiagnosticTestResult, test_id, f'/test/{test_result.results_from.single().uid}/')
