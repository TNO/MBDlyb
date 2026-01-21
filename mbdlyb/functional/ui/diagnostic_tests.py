# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""

from nicegui import ui, APIRouter
from neomodel import db
from mbdlyb.functional.gdb import Cluster, Function, Hardware, DiagnosticTest, DiagnosticTestResult
from mbdlyb.ui.helpers import get_object_or_404, get_relation_or_404, goto

from .base import Button, page, confirm_delete, confirm, TableColumn, TableMultiColumn, Table, ConditionalButton
from .helpers import save_object, save_new_object, TestsRelation
from .validation import base_name_validation

router = APIRouter()


class TestCost:
	def __init__(self, name: str, cost: float):
		self.name = name
		self.cost = cost

	def __eq__(self, other):
		if isinstance(other, TestCost):
			return self.name == other
		return False

	@classmethod
	def to_list(cls, d: dict[str, float]) -> list['TestCost']:
		return [TestCost(k, v) for k, v in d.items()]

	@classmethod
	def from_list(cls, l: list['TestCost']) -> dict[str, float]:
		return {f.name: f.cost for f in l}


def add_cost(test_costs: list[TestCost]):
	name = 'Cost_'
	for i in range(1, 100):
		if name + str(i) not in TestCost.from_list(test_costs):
			name = name + str(i)
			break
	test_costs.append(TestCost(name, 1.))
	cost_table.refresh()


def drop_cost(test_costs: list[TestCost], fault: TestCost):
	test_costs.remove(fault)
	cost_table.refresh()


@ui.refreshable
def cost_table(test_costs: list[TestCost], cost_names: list[str]):
	with ui.grid(columns=3).classes('vertical-bottom'):
		for col in ('Name', 'Cost'):
			ui.label(col).classes('font-bold')
		ui.icon('add', color='primary').on('click', lambda: add_cost(test_costs)).classes('cursor-pointer')
		for cost in test_costs:
			ui.select(cost_names, label='Name', new_value_mode='add-unique').bind_value(cost, 'name')
			ui.number('Cost', min=0., step=1.).bind_value(cost, 'cost')
			ui.icon('delete', color='negative').on('click', lambda f=cost: drop_cost(test_costs, f)).classes(
				'self-end cursor-pointer')


class TestProps:
	def __init__(self, add_result: bool = False, result_name: str = ''):
		self.add_result: bool = add_result
		self.result_name: str = result_name


def test_form(test: DiagnosticTest, test_costs: list[TestCost], test_props: TestProps, cluster: Cluster,
			  cost_names: list[str], update: bool = True):
	test_name_input = ui.input('Name', validation=base_name_validation(cluster, test if update else None)) \
		.bind_value(test, 'name').props('hide-bottom-space')
	if not update:
		ui.checkbox('Add default test result?').bind_value(test_props, 'add_result')
		test_result_input = ui.input('Result name', placeholder='Leave empty to append \'_Result\' to the test name',
		   validation=base_name_validation(cluster, allow_empty=True)).bind_visibility_from(
			   test_props, 'add_result').bind_value(test_props, 'result_name').props('hide-bottom-space')
		test_name_input.validation['Test and Result must have different names'] = lambda s: s != test_result_input.value
		test_result_input.validation['Test and Result must have different names'] = lambda s: s != test_name_input.value
		test_name_input.on_value_change(test_result_input.validate)
		test_result_input.on_value_change(test_name_input.validate)
	cost_table(test_costs, cost_names)


def _get_defined_costs() -> list[str]:
	return list(sorted({n for t in DiagnosticTest.nodes.all() for n, c in t.fixed_cost.items()}))


@router.page('/cluster/{cluster_id}/test/new/')
def test_create(cluster_id: str):
	def _save(test: DiagnosticTest, test_costs: list[TestCost], props: TestProps, cluster: Cluster):
		test.fixed_cost = TestCost.from_list(test_costs)
		save_new_object(test, cluster, disable_redirect=props.add_result)
		if props.add_result:
			result = DiagnosticTestResult(name=props.result_name or f'{test.name}_Result')
			save_new_object(result, cluster, True)
			result.results_from.connect(test)
			goto(f'/test_result/{result.uid}/tests/')

	cluster = get_object_or_404(Cluster, uid=cluster_id)
	if cluster is None:
		return
	defined_costs = _get_defined_costs()
	test = DiagnosticTest(name='', fixed_cost={n: 0. for n in defined_costs})
	test_costs = TestCost.to_list(test.fixed_cost)
	test_props = TestProps()
	page(f'New diagnostic test in {cluster.name}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(test, test_costs, test_props, cluster))
	])
	test_form(test, test_costs, test_props, cluster, defined_costs, False)


@router.page('/test/{test_id}/update/')
def test_update(test_id: str):
	def _save(test: DiagnosticTest, test_costs: list[TestCost]):
		test.fixed_cost = TestCost.from_list(test_costs)
		save_object(test, f'/cluster/{cluster.uid}/')

	test: DiagnosticTest = get_object_or_404(DiagnosticTest, uid=test_id)
	if test is None:
		return
	cluster = test.get_net()
	defined_costs = _get_defined_costs()
	test_costs = TestCost.to_list(test.fixed_cost)
	test_props = TestProps()
	page(f'Update diagnostic test {test.fqn}', cluster, [
		Button('Discard', None, 'warning', lambda: goto(f'/cluster/{cluster.uid}/')),
		Button('Save', 'save', None, lambda: _save(test, test_costs))
	])
	test_form(test, test_costs, test_props, cluster, defined_costs)


@router.page('/test/{test_id}/')
def test_details(test_id: str):
	def _move_up(test: DiagnosticTest):
		test.set_net(test.get_net().get_net())
		goto(f'/test/{test.uid}/')

	test: DiagnosticTest = get_object_or_404(DiagnosticTest, uid=test_id)
	if test is None:
		return
	cluster: Cluster = test.get_net()
	buttons = [
		Button(None, 'edit', 'primary', f'/test/{test.uid}/update/', 'Edit diagnostic test'),
		Button(None, 'delete', 'negative', lambda: confirm_delete(test, f'/cluster/{cluster.uid}/'),
			   'Delete diagnostic test')]
	if not (test.at_root or cluster.at_root):
		buttons = [Button(None, 'move_up', 'secondary', lambda: confirm('Move up test',
																		f'Are you sure to move {test.name} up to {cluster.get_net().fqn}?',
																		'I am sure', lambda: _move_up(test)),
						  'Move diagnostic test up')] + buttons
	page(f'Diagnostic test {test.name}', cluster, buttons)
	with ui.row():
		ui.label('Costs').classes('font-bold')
		ui.label(', '.join(f'{n}({c})' for n, c in test.fixed_cost.items()))
	with ui.grid(columns='1fr 1fr').classes('w-full'):
		with ui.card().classes('col-span-full'):
			Table('Test results', test.test_results.order_by('name'), [
				TableColumn('Name', 'name', None, 'fqn'),
				TableColumn('FPR', 'fp_rate', None, None),
				TableColumn('FNR', 'fn_rate', None, None),
				TableMultiColumn('Tested functions', 'name', None, 'fqn', 'indicated_functions'),
				TableMultiColumn('Tested hardware', 'name', None, 'fqn', 'indicated_hardware')
			], test, [
					  Button(icon='add', color='positive', tooltip='Add test result',
							 handler=f'/test/{test.uid}/results/new/')
				  ], [
					  Button(icon='border_all', color='secondary', tooltip='Update CPT of {}',
							 handler=lambda x: goto(f'/test_result/{x.uid}/cpt/')),
					  Button(icon='table_view', color='primary', tooltip='Edit tested items of {}',
							 handler=lambda x: goto(f'/test_result/{x.uid}/tests/')),
					  Button(icon='edit', tooltip='Edit {}', handler=lambda x: goto(f'/test_result/{x.uid}/update/')),
					  Button(icon='delete', color='negative', tooltip='Delete {}',
							 handler=lambda x: confirm_delete(x, f'/test/{test.uid}/'))
				  ]).show()


# TESTS RELATION FORM

def tests_relation_form(relation: TestsRelation, node_type: str):
	options, _ = db.cypher_query(
		f'''MATCH
    (o:DiagnosticTest {{uid: '{relation.end_uid}'}})-[:PART_OF]->(:Cluster)(()<-[:PART_OF]-()){{0,99}}(:Cluster)<-[:PART_OF]-(x:{node_type})
WHERE
    NOT (x)-[:TESTED_BY]->(o)
 RETURN x''', resolve_objects=True)
	options_dict = dict() if relation.start_uid is None else {relation.start_uid: relation.get_start_node().fqn}
	ui.select({**options_dict, **{sf[0].uid: sf[0].fqn for sf in options}}, label=node_type,
			  with_input=True).bind_value(
		relation, 'start_uid')
	ui.number('Weight', min=0., max=1., precision=2, step=.01).bind_value(relation, 'weight')


# TESTS FUNCTION RELATION

@router.page('/test/{test_id}/tests_fn/new/')
def tests_fn_relation_create(test_id: str):
	test = get_object_or_404(DiagnosticTest, uid=test_id)
	if test is None:
		return

	relation = TestsRelation(end_node=test, node_type=Function)
	page(f'Add tested function to {test}', test.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/test/{test.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/test/{test.uid}/'))
	])
	tests_relation_form(relation, 'Function')


@router.page('/test/{test_id}/tests_fn/{function_id}/update/')
def tests_fn_relation_update(test_id: str, function_id: str):
	test = get_object_or_404(DiagnosticTest, uid=test_id)
	function = get_object_or_404(Function, uid=function_id)
	relation = get_relation_or_404(test, function, 'tested_functions')
	if None in (test, function, relation):
		return

	relation = TestsRelation(function, test, weight=relation.weight)
	page(f'Update {relation}', test.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/test/{test.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/test/{test.uid}/'))
	])
	tests_relation_form(relation, 'Function')


# TESTS HARDWARE RELATION

@router.page('/test/{test_id}/tests_hw/new/')
def tests_hw_relation_create(test_id: str):
	test = get_object_or_404(DiagnosticTest, uid=test_id)
	if test is None:
		return

	relation = TestsRelation(end_node=test, node_type=Hardware)
	page(f'Add tested hardware to {test}', test.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/test/{test.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/test/{test.uid}/'))
	])
	tests_relation_form(relation, 'Hardware')


@router.page('/test/{test_id}/tests_hw/{hardware_id}/update/')
def observes_hw_relation_update(test_id: str, hardware_id: str):
	test = get_object_or_404(DiagnosticTest, uid=test_id)
	hardware = get_object_or_404(Hardware, uid=hardware_id)
	relation = get_relation_or_404(test, hardware, 'tested_hardware')
	if None in (test, hardware, relation):
		return

	relation = TestsRelation(hardware, test, weight=relation.weight)
	page(f'Update {relation}', test.get_net(), [
		Button('Discard', None, 'warning', lambda: goto(f'/test/{test.uid}/')),
		Button('Save', 'save', None, lambda: relation.save(f'/test/{test.uid}/'))
	])
	tests_relation_form(relation, 'Hardware')
