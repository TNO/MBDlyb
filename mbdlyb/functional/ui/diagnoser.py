# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import io
import re
import asyncio
from typing import Union, Tuple

import pandas as pd
import plotly.express as px
from nicegui import app, ui, APIRouter, run
from nicegui.elements.mixins.value_element import ValueElement
from nicegui.elements.slider import Slider
from nicegui.events import ValueChangeEventArguments

from mbdlyb.functional import (gdb, Diagnoser, Cluster, DiagnosticTest, DirectObservable, OperatingMode, FunctionalNode,
							   DiagnosticTestResult)
from mbdlyb.ui.utils import Status, status, show_name

from .base import header, footer, goto, reasoner_class_dict, confirm
from ... import longest_common_fqn

router = APIRouter(prefix='/cluster/{cluster_id}')


class UIDiagnoserData:
	_cluster_id: str = None
	cluster: Cluster = None
	diagnoser: Diagnoser = None

	_staging_evidence: set[str] = None
	_unstaging_evidence: set[str] = None

	health_results: pd.DataFrame = None
	diagnostic_tests: pd.DataFrame = None
	direct_observables: pd.DataFrame = None
	operating_modes: pd.DataFrame = None

	filter_hw: str = None
	filter_dt: str = None
	filter_do: str = None
	filter_opm: str = None

	hardware_limit: int = 10
	test_limit: int = 10
	entropy_limit: int = 10

	auto_compute: bool = True
	soft_evidence: bool = False
	balance: float = 0.5

	def __init__(self, cluster_id: str, auto_compute=False):
		self._cluster_id = cluster_id
		self.auto_compute = auto_compute
		self.empty()

	def empty(self):
		self._empty_staging_evidence()
		self._empty_unstaging_evidence()
		self.health_results = pd.DataFrame()
		self.diagnostic_tests = pd.DataFrame()
		self.direct_observables = pd.DataFrame()
		self.operating_modes = pd.DataFrame()
		if self.diagnoser:
			self.diagnoser.reasoner.drop_evidence()
			self.diagnoser.reasoner.drop_targets()

	async def reset(self, compute_button, event=None):
		status.refresh(Status.COMPUTING)
		if event is not None:
			event.sender.disable()
		self.empty()
		self.refresh(True)

		reasoner_klass_name = app.storage.general.get('reasoner')
		reasoner_klass = reasoner_class_dict[reasoner_klass_name]

		self.cluster = await run.io_bound(gdb.Cluster.load, self._cluster_id)
		self.diagnoser = await run.io_bound(Diagnoser, self.cluster, reasoner_klass)
		self.diagnoser.set_entropy_limit(self.entropy_limit)
		show_name.refresh(self.cluster.name)

		if event is not None:
			event.sender.enable()
		compute_button.run_method('click', {'compute_diagnoses': self.auto_compute})

	async def compute(self, event=None, args={}):
		compute_diagnoses = args.get('compute_diagnoses', True)
		status.refresh(Status.COMPUTING)
		if event is not None:
			event.sender.disable()

		self.diagnoser.set_loss_function(lambda e, c: self.balance * e + (1-self.balance) * c)
		if compute_diagnoses:
			if self._unstaging_evidence:
				self.diagnoser.drop_evidence(*self._unstaging_evidence)
			self.health_results = await run.io_bound(self.diagnoser.infer)
			self.health_results.sort_values(by="Healthy", inplace=True)
			self.diagnostic_tests = await run.io_bound(self.diagnoser.compute_next_diagnostic_tests)
			self._empty_staging_evidence()
			self._empty_unstaging_evidence()
		self.direct_observables = await run.io_bound(self.diagnoser.get_direct_observables)
		self.operating_modes = await run.io_bound(self.diagnoser.get_operating_modes)
		self.refresh(compute_diagnoses)
		if event is not None:
			event.sender.enable()
		status.refresh(Status.READY)

	async def compute_tests(self, event=None):
		status.refresh(Status.COMPUTING)
		if event is not None:
			event.sender.disable()

		self.diagnoser.set_loss_function(lambda e, c: self.balance * e + (1 - self.balance) * c)
		self.diagnostic_tests = await run.io_bound(self.diagnoser.sort_next_diagnostic_tests)
		diagnostic_tests.refresh()
		diagnostic_tests_plot.refresh()
		if event is not None:
			event.sender.enable()
		status.refresh(Status.READY)

	@property
	def staging_evidence(self) -> set[str]:
		return self._staging_evidence

	def add_staging_evidence(self, *fqns: str):
		for fqn in fqns:
			self._staging_evidence.add(fqn)

	def _empty_staging_evidence(self):
		self._staging_evidence = set()

	@property
	def unstaging_evidence(self) -> set[str]:
		return self._unstaging_evidence

	def add_unstaging_evidence(self, *fqns: str):
		for fqn in fqns:
			self._unstaging_evidence.add(fqn)
			if fqn in self._staging_evidence:
				self._staging_evidence.remove(fqn)

	def _empty_unstaging_evidence(self):
		self._unstaging_evidence = set()

	@staticmethod
	def refresh(refresh_diagnoses=True):
		if refresh_diagnoses:
			diagnoses.refresh()
			diagnostic_tests.refresh()
			diagnostic_tests_plot.refresh()
		direct_observables.refresh()
		operating_modes.refresh()
		evidence.refresh()


@router.page('/diagnoser/')
def diagnoser_main(cluster_id: str):
	auto_compute = app.storage.general.get('auto_compute', False)
	data: UIDiagnoserData = UIDiagnoserData(cluster_id, auto_compute)

	header('Functional Diagnoser')
	footer()

	with ui.row().classes('w-full gap-5 items-center'):
		show_name('Loading...')
		status(Status.COMPUTING)
		compute_button = ui.button('Compute', icon='refresh', color='primary').on('click', lambda e: data.compute(e, e.args))
		ui.checkbox('Auto-compute').bind_value(data, 'auto_compute')
		ui.checkbox('Soft evidence').bind_value(data, 'soft_evidence').on_value_change(lambda: (diagnostic_tests.refresh(), direct_observables.refresh()))
		ui.label('Balance: (cost vs entropy)')
		slider = ui.slider(min=0.0, max=1.0, step=0.01, value=0.5).bind_value(data, 'balance').style('width: 200px').on_value_change(data.compute_tests)
		ui.label().bind_text_from(slider, 'value', lambda v: f'{v:.2f}')
		ui.space()
		ui.button('Reset', icon='replay', color='negative').on('click', lambda e: confirm('Reset diagnoser',
																						   'Are you sure you want to reset the diagnoser?',
																						   'Reset', (data.reset,
																									 (compute_button,
																									  e)), 'negative'))
		ui.button('Editor', icon='home', color='primary').on_click(lambda: goto(f'/cluster/{cluster_id}'))

	box_classes = 'p-2 border border-gray-300 rounded-borders h-full'
	with ui.grid(columns='1fr 1fr').classes('fit items-stretch gap-1'):
		with ui.column().classes(box_classes):
			with ui.row().classes('w-full items-center'):
				ui.label('Diagnoses').classes('text-h5')
				ui.space()
				ui.input(label='Filter').bind_value(data, 'filter_hw').props('clearable flat dense').on_value_change(
					lambda x: diagnoses.refresh())
			diagnoses(data)
		with ui.column().classes(box_classes):
			dialog = ui.dialog().props('full-width full-height')
			diagnostic_tests_plot(data, dialog)
			with ui.row().classes('w-full items-center'):
				ui.label('Diagnostic tests').classes('text-h5')
				ui.space()
				ui.input(label='Filter').bind_value(data, 'filter_dt').props('clearable flat dense').on_value_change(
					lambda x: diagnostic_tests.refresh())
				ui.button(icon='insert_chart', on_click=dialog.open).props('flat').tooltip('Plot diagnostic tests')
			diagnostic_tests(data)
	with ui.grid(columns='1fr 1fr').classes('fit items-stretch gap-1'):
		with ui.column().classes(box_classes):
			with ui.row().classes('w-full items-center'):
				ui.label('Operating modes').classes('text-h5')
				ui.space()
				ui.input(label='Filter').bind_value(data, 'filter_opm').props('clearable flat dense').on_value_change(
					lambda x: operating_modes.refresh())
			operating_modes(data)
		with ui.column().classes(box_classes):
			with ui.row().classes('w-full items-center'):
				ui.label('Direct observables').classes('text-h5')
				ui.space()
				ui.input(label='Filter').bind_value(data, 'filter_do').props('clearable flat dense').on_value_change(
					lambda x: direct_observables.refresh())
			direct_observables(data)
	with ui.grid(columns='1fr').classes('fit items-stretch gap-1'):
		with ui.column().classes(box_classes):
			with ui.row().classes('w-full items-center'):
				ui.label('Evidence').classes('text-h5')
				ui.space()
				ui.button(icon='download', on_click=lambda: ui.download(export(data), f'results_{data.cluster.name}.xlsx')).props('flat').tooltip('Download evidence')
			evidence(data)

	asyncio.create_task(data.reset(compute_button))


def _normalize_sliders(sliders: list[Slider]):
	if all(sliders):
		values = [s.value for s in sliders]
		_sum = sum(values)
		if _sum <= 0.:
			_count = len(sliders)
			return [1. / _count for _ in sliders]
		return [v / sum(values) for v in values]
	else:
		return [0] * len(sliders)


def _check_on_value_change(e: ValueChangeEventArguments, evidence: dict, fqn: str, sliders: list[Slider]):
	if not e.value:
		del evidence[fqn]
	else:
		evidence[fqn] = [sliders[0].value, 1. - sliders[0].value] if len(sliders) == 1 else _normalize_sliders(sliders)


def create_evidence_select(label: str, states: list[str],
						   evidence: dict[str, Union[str, list[float]]], fqn: str) -> list[ValueElement]:
	return [ui.select(states, label=label).bind_value(evidence, fqn).classes('w-64')]


def create_evidence_sliders(states: list[str], evidence: dict[str, Union[str, list[float]]], fqn: str,
							make_optional: bool = True, label: str = None) -> list[
	Union[ValueElement, list[ValueElement], Tuple[ValueElement, Union[ValueElement, list[ValueElement]]]]]:
	# In case there are exactly two states: create one slider which balances between the two
	if len(states) == 2:
		with ((ui.row().classes('items-center'))):
			if label:
				with ui.row().classes('w-full'):
					if make_optional:
						checkbox = ui.checkbox(label, value=True)
					else:
						ui.label(label)
			with ui.row().classes('items-center'):
				if make_optional and not label:
					checkbox = ui.checkbox(value=True)
				ui.label(states[1])
				slider = ui.slider(min=0.0, max=1.0, step=0.01, value=0.5).style('width: 200px')
				slider.bind_value_to(evidence, fqn, forward=lambda v: [v, 1.-v])
				if make_optional:
					slider.bind_enabled_from(checkbox, 'value')
					checkbox.on_value_change(lambda e: _check_on_value_change(e, evidence, fqn, [slider]))
				ui.label(states[0])
				ui.label().bind_text_from(slider, 'value', lambda v: f'{v:.2f}')
			return [(checkbox, slider)] if make_optional else [slider]
	# In any other case: create a slider per state and normalize between the sliders' values
	else:
		sliders: list[Slider] = []
		if make_optional or label:
			with ui.row().classes('w-full items-center'):
				if make_optional:
					checkbox = ui.checkbox(text=label or '', value=True)
				elif label:
					ui.label(label)
		for i, state in enumerate(states):
			with ui.row().classes('items-center'):
				ui.label(state).classes('w-16')
				slider = ui.slider(min=0.0, max=1.0, step=0.01, value=0.5).style('width: 200px')
				slider.bind_value_to(evidence, fqn, forward=lambda _: _normalize_sliders(sliders))
				sliders.append(slider)
				ui.label().bind_text_from(sliders[i], 'value', lambda v: f'{v:.2f}')
		if make_optional:
			for slider in sliders:
				slider.bind_enabled_from(checkbox, 'value')
			checkbox.on_value_change(lambda e: _check_on_value_change(e, evidence, fqn, sliders))
		return [(checkbox, sliders)] if make_optional else [sliders]


@ui.refreshable
def diagnoses(data: UIDiagnoserData):
	filter_expression = f'.*{data.filter_hw}.*' if data.filter_hw else None
	f_d_data = data.health_results if filter_expression is None else data.health_results[
		data.health_results.index.map(
			lambda x: re.search(filter_expression, x.fqn, re.IGNORECASE) is not None)]

	_diagnoses = f_d_data.iloc[:data.hardware_limit]
	if _diagnoses.size == 0:
		ui.label('No results to show.').classes('text-grey')
		return
	lc_fqn = longest_common_fqn(*_diagnoses.index)
	diagnoses_index = [re.sub(f'^{lc_fqn}\\.*', '', idx.fqn if idx.name != idx.net.name else idx.net.fqn) or idx.name
					   for idx in _diagnoses.index]
	data = {
		'data': [{
			'type': 'bar',
			'name': m,
			'orientation': 'h',
			'marker': {
				'color': '#99CC99' if m == 'Healthy' else '#CC9999'
			},
			'x': [round(d, 3) for d in _diagnoses[m]],
			'y': diagnoses_index,
			'hovertemplate': '%{x}',
		} for m in _diagnoses.columns],
		'layout': {
			'barmode': 'stack',
			'showlegend': False,
			'margin': {
				'pad': 5,
				't': 0,
				'b': 0,
				'r': 0,
				'l': 0
			},
			'yaxis': {
				'automargin': True, 
				'autorange': 'reversed',
				'tickfont': {
					'size': 14
				}
			}
		},
		'config': {
			'responsive': True
		}
	}
	ui.plotly(data).classes('w-full')

def _tuple_has_value(t: Tuple[ValueElement, Union[ValueElement, list[ValueElement]]]) -> bool:
	checkbox, sliders = t
	return checkbox.value and all(_value_element_has_value(s) for s in sliders)

def _value_element_has_value(ve: Union[ValueElement, list[ValueElement]]) -> bool:
	if isinstance(ve, list):
		return all(_value_element_has_value(_ve) for _ve in ve)
	return ve.value >= 0. if isinstance(ve.value, float) else ve.value is not None


def _check_evidence_provided(inputs: list[
	Union[ValueElement, list[ValueElement], Tuple[ValueElement, Union[ValueElement, list[ValueElement]]]]]) -> bool:
	return any(_tuple_has_value(x) if isinstance(x, tuple) else _value_element_has_value(x) for x in inputs)


@ui.refreshable
def diagnostic_tests(data: UIDiagnoserData):
	filter_expression = f'.*{data.filter_dt}.*' if data.filter_dt else None
	f_dt_data = data.diagnostic_tests if filter_expression is None else data.diagnostic_tests[
		data.diagnostic_tests['Diagnostic Test'].map(
			lambda x: re.search(filter_expression, x.fqn, re.IGNORECASE) is not None)]

	if f_dt_data.size == 0:
		ui.label('No tests to show.').classes('text-grey')
		return

	async def _save(dt, dt_ev, exp):
		data.add_staging_evidence(*dt_ev.keys())
		data.diagnoser.perform_diagnostic_test(dt, dt_ev)
		evidence.refresh()
		exp.close()
		exp.delete()
		if data.auto_compute:
			await data.compute()

	lc_fqn = longest_common_fqn(*f_dt_data['Diagnostic Test'].map(lambda x: x.fqn))
	dt_ev: dict[DiagnosticTest, dict[str, str]] = dict()
	with ui.scroll_area().classes('fit'):
		for _, r in f_dt_data.iterrows():
			dt: DiagnosticTest = r['Diagnostic Test']
			dt_ev[dt] = dict()
			with ui.expansion(re.sub(f'^{lc_fqn}\\.*', '', dt.fqn) or dt.name, caption=f'Cost: {r["Cost"]:.2f}, Entropy: {r["Entropy"]:.3f}').classes('w-full') as exp:
				inputs: list[Union[ValueElement, Tuple[ValueElement, list[ValueElement], Union[ValueElement, list[ValueElement]]]]] = []
				for test_result in dt.test_results:
					if data.soft_evidence:
						inputs += create_evidence_sliders(test_result.states, dt_ev[dt], test_result.fqn, label=test_result.name)
					else:
						inputs += create_evidence_select(test_result.name, test_result.states, dt_ev[dt], test_result.fqn)
				btn = ui.button('Save', on_click=lambda t=dt, ev=dt_ev[dt], e=exp: _save(t, ev, e))
				btn.bind_enabled_from(locals(), 'inputs', backward=_check_evidence_provided)


@ui.refreshable
def diagnostic_tests_plot(data: UIDiagnoserData, dialog: ui.dialog):
	dialog.clear()
	if data.diagnostic_tests.size == 0:
		return

	# Update plotly's blue colorscale with transparency
	blues = px.colors.get_colorscale('blues')
	for i, color in enumerate(blues):
		blues[i][1] = re.sub(r'rgb\((\d*),(\d*),(\d*)\)', r'rgba(\1,\2,\3,0.33)', color[1])

	with dialog, ui.card():
		contour_x, contour_y, contour_z = data.diagnoser.compute_loss_grid()
		plot_data = {
			'data': [
				{
					'type': 'scatter',
					'mode': 'markers+text',
					'name': 'Cost vs Entropy',
					'x': [round(d, 3) for d in data.diagnostic_tests['Entropy']],
					'y': [round(d, 2) for d in data.diagnostic_tests['Cost']],
					'hovertext': [dt.fqn for dt in data.diagnostic_tests['Diagnostic Test']],
				},
				{
					'type': 'contour',
					'x': contour_x,
					'y': contour_y,
					'z': contour_z,
					'hoverinfo': 'skip',
					'colorscale': blues,
					'showscale': False,
					'autocontour': False,
					'contours': {
    					'start': 0,
    					'end': 1,
						'size': 0.1,
						'coloring': 'heatmap',
					},
					'zmin': 0,
					'zmax': 1,
				}
			],
			'layout': {
				'showlegend': False,
				'margin': {
					'pad': 5
				},
				'title': 'Cost vs Entropy',
				'xaxis': {
					'title': 'Entropy',
					'autorange': False,
					'range': [contour_x.min(), contour_x.max()],
				},
				'yaxis': {
					'title': 'Cost',
					'autorange': False,
					'range': [contour_y.min(), contour_y.max()],
				},
				'zaxis': {
					'autorange': False,
					'range': [0.0, 1.0],
				},
				'annotations': [
					{'x': r['Entropy'],
					 'y': r['Cost'],
					 'text': r['Diagnostic Test'].name,
					 'textangle': -45,
					 'showarrow': False,
					 'xanchor': 'left',
					 'yanchor': 'bottom',}
				for _, r in data.diagnostic_tests.iterrows()],
			},
			'config': {
				'responsive': True
			}
		}
		ui.plotly(plot_data).classes('fit')
		with ui.row().classes('w-full justify-center'):
			ui.button('Close', on_click=dialog.close)


@ui.refreshable
def direct_observables(data: UIDiagnoserData):
	filter_expression = f'.*{data.filter_do}.*' if data.filter_do else None
	f_do_data = data.direct_observables if filter_expression is None else data.direct_observables[
		data.direct_observables['Direct Observable'].map(
			lambda x: re.search(filter_expression, x.fqn, re.IGNORECASE) is not None)]

	if data.direct_observables.size == 0:
		ui.label('No results to show.').classes('text-grey')
		return

	async def _save(dt, dt_ev, exp):
		data.add_staging_evidence(*dt_ev.keys())
		data.diagnoser.add_evidence(dt_ev)
		evidence.refresh()
		exp.close()
		exp.delete()
		if data.auto_compute:
			await data.compute()

	lc_fqn = longest_common_fqn(*f_do_data['Direct Observable'].map(lambda x: x.fqn))
	with ui.scroll_area().classes('w-full'):
		for _, r in f_do_data.iterrows():
			obs: DirectObservable = r['Direct Observable']
			ev = dict()
			with ui.expansion(re.sub(f'^{lc_fqn}\\.*', '', obs.fqn) or obs.name).classes('w-full') as exp:
				inputs: list[Union[ValueElement, Tuple[ValueElement, Union[ValueElement, list[ValueElement]]]]] = []
				if data.soft_evidence:
					inputs = create_evidence_sliders(obs.states, ev, obs.fqn, make_optional=False)
				else:
					inputs = create_evidence_select('Result', obs.states, ev, obs.fqn)
				btn = ui.button('Save', on_click=lambda t=obs, evd=ev, e=exp: _save(t, evd, e))
				btn.bind_enabled_from(locals(), 'inputs', backward=_check_evidence_provided)


@ui.refreshable
def operating_modes(data: UIDiagnoserData):
	filter_expression = f'.*{data.filter_opm}.*' if data.filter_opm else None
	f_opm_data = data.operating_modes if filter_expression is None else data.operating_modes[
		data.operating_modes['Operating mode'].map(
			lambda x: re.search(filter_expression, x.fqn, re.IGNORECASE) is not None)]

	if data.operating_modes.size == 0:
		ui.label('No operating modes can be selected.').classes('text-grey')
		return

	async def _save(opm_ev, exp):
		data.add_staging_evidence(*opm_ev.keys())
		data.diagnoser.add_operating_mode(opm_ev)
		evidence.refresh()
		exp.close()
		exp.delete()
		if data.auto_compute:
			await data.compute()

	lc_fqn = longest_common_fqn(*f_opm_data['Operating mode'].map(lambda x: x.fqn))
	with ui.scroll_area().classes('w-full'):
		for _, r in f_opm_data.iterrows():
			opm: OperatingMode = r['Operating mode']
			ev = dict()
			with ui.expansion(re.sub(f'^{lc_fqn}\\.*', '', opm.fqn) or opm.name).classes('w-full') as exp:
				sel = create_evidence_select('Operating mode', opm.states, ev, opm.fqn)[0]
				ui.button('Save', on_click=lambda evd=ev, e=exp: _save(evd, e)).bind_enabled_from(sel, 'value')


def export(data: UIDiagnoserData) -> bytes:
	bytes = io.BytesIO()
	try:
		evidence = [[fqn, *ev] for fqn, ev in data.diagnoser.reasoner.evidence.items()]
		evidence_df = pd.DataFrame(evidence, columns=['Test', 'Ok', 'NOk'])

		# get the health of each hardware node
		health_df = data.health_results.rename_axis('Hardware').reset_index()

		# get the list of recommended tests - sort by entropy and name
		columns = ['Diagnostic Test', 'Entropy', 'Cost']
		balance = data.balance
		balance_df = pd.DataFrame([['', '', ''], ['Balance', balance, '']], columns=columns)
		tests_df = data.diagnostic_tests[columns]
		tests_df = pd.concat([tests_df, balance_df], ignore_index=True, sort=False)

		with pd.ExcelWriter(bytes) as writer:
			evidence_df.to_excel(writer, sheet_name='evidence', index=False)
			health_df.to_excel(writer, sheet_name='health', index=False)
			tests_df.to_excel(writer, sheet_name='tests', index=False)
		bytes.seek(0,0)
	except Exception as e:
		pass
	return bytes.read()


@ui.refreshable
def evidence(data: UIDiagnoserData):
	async def _delete_evidence(d: UIDiagnoserData, node: FunctionalNode):
		fqns = [n.fqn for n in node.test.test_results] if isinstance(node, DiagnosticTestResult) else [node.fqn]
		d.add_unstaging_evidence(*fqns)
		evidence.refresh()
		if data.auto_compute:
			await data.compute()

	if data.diagnoser is None or not data.diagnoser.reasoner.evidence:
		ui.label('No evidence to show.').classes('text-grey')
		return

	with (ui.scroll_area().classes('w-full')):
		border_classes = 'border-b border-b-gray-300'
		def_classes = f'p-2 {border_classes}'
		with ui.grid(columns='40px auto 1fr 40px').classes('w-full items-center gap-0'):
			ui.html('&nbsp;', sanitize=False).classes('text-h6').classes(def_classes)
			ui.label('Result').classes('text-h6').classes(def_classes)
			ui.label('Evidence').classes('text-h6').classes(def_classes)
			ui.html('&nbsp;', sanitize=False).classes('text-h6').classes(def_classes)

			for fqn, ev in reversed(data.diagnoser.reasoner.evidence.items()):
				node = data.diagnoser.net.get_node(fqn)
				deletable = True
				if isinstance(node, OperatingMode):
					index = ev.index(1.)
					_status = node.states[index]
					deletable = False
				elif ev == [0.0, 1.0]:
					_status = '❌'
				elif ev == [1.0, 0.0]:
					_status = '✔️'
				else:
					_status = '[' + ', '.join(f'{e:.2f}' for e in ev) + ']'

				with ui.row().classes(f'h-full items-center justify-evenly {border_classes}'):
					if fqn in data.staging_evidence:
						ui.icon('playlist_add').classes('text-gray').tooltip('Recently added evidence, yet to be included in the diagnosis.')
					elif fqn in data.unstaging_evidence:
						ui.icon('delete').classes('text-negative').tooltip('Evidence marked for deletion upon computation of next diagnosis.')
					else:
						ui.icon('check_circle').classes('text-positive').tooltip('Included in diagnosis.')
				ui.label(fqn).classes(def_classes)
				ui.label(_status).classes(def_classes)
				with ui.row().classes(f'h-full items-center justify-evenly {border_classes}'):
					if deletable:
						ui.button(icon='delete', on_click=lambda d=data, n=node: confirm('Remove evidence',
																						 f'Are you sure you want to delete the provided evidence for test {n.test}? All test results associated to the test will be deleted.' if isinstance(
																							 n,
																							 DiagnosticTestResult) else f'Are you sure you want to delete the provided evidence for {n}?',
																						 'Delete', lambda _d=d,
																										  _n=n: _delete_evidence(
								_d, _n), confirm_color='negative')).classes('text-negative').props(
							'flat padding="none"')
