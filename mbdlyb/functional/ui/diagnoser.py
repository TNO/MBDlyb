# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import io
import re

import pandas as pd
import plotly.express as px
from nicegui import app, ui, APIRouter, run

from mbdlyb.functional import gdb, Diagnoser, Cluster, DiagnosticTest, DirectObservable, OperatingMode
from mbdlyb.ui.utils import Status, status, show_name

from .base import header, footer, goto


router = APIRouter(prefix='/cluster/{cluster_id}')


class UIDiagnoserData:
	_cluster_id: str = None
	cluster: Cluster = None
	diagnoser: Diagnoser = None

	health_results: pd.DataFrame = None
	test_results: pd.DataFrame = None
	direct_observables: pd.DataFrame = None
	operating_modes: pd.DataFrame = None

	hardware_limit: int = 10
	test_limit: int = 10
	entropy_limit: int = 10

	auto_compute: bool = True
	soft_evidence: bool = False
	balance: float = 0.5

	def __init__(self, cluster_id: str, auto_compute=False):
		self._cluster_id = cluster_id
		self.health_results = pd.DataFrame()
		self.test_results = pd.DataFrame()
		self.direct_observables = pd.DataFrame()
		self.operating_modes = pd.DataFrame()
		self.auto_compute = auto_compute

	def reset(self):
		self.cluster = gdb.Cluster.load(self._cluster_id)
		self.diagnoser = Diagnoser(self.cluster)
		self.diagnoser.set_entropy_limit(self.entropy_limit)
		show_name.refresh(self.cluster.name)
		status.refresh(Status.READY)

	async def compute(self, event=None, args={}):
		compute_diagnoses = args.get('compute_diagnoses', True)
		status.refresh(Status.COMPUTING)
		if event is not None:
			event.sender.disable()

		self.diagnoser.set_loss_function(lambda e, c: self.balance * e + (1-self.balance) * c)
		if compute_diagnoses:
			self.health_results = await run.io_bound(self.diagnoser.infer)
			self.health_results.sort_values(by="Healthy", inplace=True)
			self.test_results = await run.io_bound(self.diagnoser.compute_next_diagnostic_tests)
		self.direct_observables = await run.io_bound(self.diagnoser.get_direct_observables)
		self.operating_modes = await run.io_bound(self.diagnoser.get_operating_modes)
		if compute_diagnoses:
			diagnoses.refresh(self.health_results.iloc[:self.hardware_limit])
			diagnostic_tests.refresh()
			diagnostic_tests_plot.refresh()
		direct_observables.refresh()
		operating_modes.refresh()
		evidence.refresh()
		if event is not None:
			event.sender.enable()
		status.refresh(Status.READY)

	async def compute_tests(self, event=None):
		status.refresh(Status.COMPUTING)
		if event is not None:
			event.sender.disable()
		
		self.diagnoser.set_loss_function(lambda e, c: self.balance * e + (1-self.balance) * c)
		self.test_results = await run.io_bound(self.diagnoser.sort_next_diagnostic_tests)
		diagnostic_tests.refresh()
		diagnostic_tests_plot.refresh()
		if event is not None:
			event.sender.enable()
		status.refresh(Status.READY)


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
		ui.button('Editor', icon='home', color='primary').on_click(lambda: goto(f'/cluster/{cluster_id}'))

	with ui.grid(columns='1fr 1fr').classes('fit items-stretch gap-1'):
		with ui.column().classes('border rounded-borders p-2 h-full'):
			ui.label('Diagnoses').classes('text-h5')
			diagnoses(data.health_results)
		with ui.column().classes('border rounded-borders p-2 h-full'):
			dialog = ui.dialog().props('full-width full-height')
			diagnostic_tests_plot(data, dialog)
			with ui.row().classes('w-full'):
				ui.label('Diagnostic tests').classes('text-h5')
				ui.space()
				ui.button('Show plot', icon='insert_chart', on_click=dialog.open)
			diagnostic_tests(data)
	with ui.grid(columns='1fr 1fr').classes('fit items-stretch gap-1'):
		with ui.column().classes('border rounded-borders p-2 h-full'):
			ui.label('Operating modes').classes('text-h5')
			operating_modes(data)
		with ui.column().classes('border rounded-borders p-2 h-full'):
			ui.label('Direct observables').classes('text-h5')
			direct_observables(data)
	with ui.grid(columns='1fr').classes('fit items-stretch gap-1'):
		with ui.column().classes('border rounded-borders p-2 h-full'):
			ui.label('Evidence').classes('text-h5')
			evidence(data)

	data.reset()
	if auto_compute:
		compute_button.run_method('click', {'compute_diagnoses': True})
	else:
		compute_button.run_method('click', {'compute_diagnoses': False})


def create_softevidence_sliders(states: list, evidence: dict, fqn: str):
	# In case there are exactly two states: create one slider which balances between the two
	if len(states) == 2:
		with ui.row().classes('items-center'):
			ui.label(states[1])
			slider = ui.slider(min=0.0, max=1.0, step=0.01, value=0.5).bind_value_to(evidence, fqn, forward=lambda v: [v, 1.-v]).style('width: 200px')
			ui.label(states[0])
			ui.space()
			ui.label().bind_text_from(slider, 'value', lambda v: f'{v:.2f}')
	# In any other case: create a slider per state and normalize between the sliders' values
	else:
		def normalize(sliders):
			if all(sliders):
				values = [s.value for s in sliders]
				return [v / sum(values) for v in values]
			else:
				return [0] * len(sliders)

		sliders = [None] * len(states)
		for i, state in enumerate(states):
			with ui.row().classes('items-center w-full'):
				ui.label(state).classes('w-16')
				sliders[i] = ui.slider(min=0.0, max=1.0, step=0.01, value=0.5).bind_value_to(evidence, fqn, forward=lambda _: normalize(sliders)).style('width: 200px')
				ui.label().bind_text_from(sliders[i], 'value', lambda v: f'{v:.2f}')


@ui.refreshable
def diagnoses(diagnoses: pd.DataFrame):
	if diagnoses.size == 0:
		ui.label('No results to show yet.').classes('text-grey')
		return
	diagnoses_index = [i.fqn for i in diagnoses.index]
	data = {
		'data': [{
			'type': 'bar',
			'name': m,
			'orientation': 'h',
			'marker': {
				'color': '#99CC99' if m == 'Healthy' else '#CC9999'
			},
			'x': [round(d, 3) for d in diagnoses[m]],
			'y': diagnoses_index,
		} for m in diagnoses.columns],
		'layout': {
			'barmode': 'stack',
			'showlegend': False,
			'margin': {
				'pad': 5	
			},
			'yaxis': {
				'automargin': True, 
				'autorange': 'reversed'
			}
		},
		'config': {
			'responsive': True
		}
	}
	ui.plotly(data).classes('w-full')


@ui.refreshable
def diagnostic_tests(data: UIDiagnoserData):
	if data.test_results.size == 0:
		ui.label('No results to show yet.').classes('text-grey')
		return

	async def _save(dt, dt_ev, exp):
		data.diagnoser.perform_diagnostic_test(dt, dt_ev)
		if data.auto_compute:
			await data.compute()
		else:
			exp.close()
			exp.delete()

	dt_ev: dict[DiagnosticTest, dict[str, str]] = dict()
	with ui.scroll_area().classes('fit'):
		for _, r in data.test_results.iterrows():
			dt: DiagnosticTest = r['Diagnostic Test']
			dt_ev[dt] = dict()
			with ui.expansion(dt.fqn, caption=f'Cost: {r["Cost"]:.2f}, Entropy: {r["Entropy"]:.3f}').classes('w-full') as exp:
				for test_result in dt.test_results:
					if data.soft_evidence:
						create_softevidence_sliders(test_result.states, dt_ev[dt], test_result.fqn)
					else:
						ui.select(test_result.states, label=test_result.name).bind_value(dt_ev[dt], test_result.fqn).classes('w-64')
				ui.button('Save', on_click=lambda t=dt, ev=dt_ev[dt], e=exp: _save(t, ev, e))


@ui.refreshable
def diagnostic_tests_plot(data: UIDiagnoserData, dialog: ui.dialog):
	dialog.clear()
	if data.test_results.size == 0:
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
					'x': [round(d, 3) for d in data.test_results['Entropy']],
					'y': [round(d, 2) for d in data.test_results['Cost']],
					'hovertext': [dt.fqn for dt in data.test_results['Diagnostic Test']],
				},
				{
					'type': 'contour',
					'x': contour_x,
					'y': contour_y,
					'z': contour_z,
					'hoverinfo': 'skip',
					'colorscale': blues,
					#'reversescale': True,
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
				for _, r in data.test_results.iterrows()],
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
	if data.direct_observables.size == 0:
		ui.label('No results to show yet.').classes('text-grey')
		return

	async def _save(dt, dt_ev, exp):
		data.diagnoser.add_evidence(dt_ev)
		if data.auto_compute:
			await data.compute()
		else:
			exp.close()
			exp.delete()

	with ui.scroll_area().classes('w-full'):
		for _, r in data.direct_observables.iterrows():
			obs: DirectObservable = r['Direct Observable']
			ev = dict()
			with ui.expansion(obs.fqn).classes('w-full') as exp:
				if data.soft_evidence:
					create_softevidence_sliders(obs.states, ev, obs.fqn)
				else:
					ui.select(obs.states, label='Result').bind_value(ev, obs.fqn).classes('w-64')
				ui.button('Save', on_click=lambda t=obs, evd=ev, e=exp: _save(t, evd, e))


@ui.refreshable
def operating_modes(data: UIDiagnoserData):
	if data.operating_modes.size == 0:
		ui.label('No operating modes can be selected.').classes('text-grey')
		return

	async def _save(dt, dt_ev, exp):
		data.diagnoser.add_operating_mode(dt_ev)
		if data.auto_compute:
			await data.compute()
		else:
			exp.close()
			exp.delete()

	with ui.scroll_area().classes('w-full'):
		for _, r in data.operating_modes.iterrows():
			opm: OperatingMode = r['Operating mode']
			ev = dict()
			with ui.expansion(opm.fqn).classes('w-full') as exp:
				ui.select(opm.states, label='Operating mode').bind_value(ev, opm.fqn).classes('w-64')
				ui.button('Save', on_click=lambda t=opm, evd=ev, e=exp: _save(t, evd, e))


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
		tests_df = data.test_results[columns]
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
	if data.diagnoser is None:
		ui.label('No results to show yet.').classes('text-grey')
		return

	columns = [
		{'name': 'Result', 'label': 'Result', 'field': 'result', 'required': True, 'align': 'left'},
		{'name': 'Status', 'label': 'Status', 'field': 'status', 'required': True, 'align': 'left'},
	]
	rows = []
	for fqn, evidence in data.diagnoser.reasoner.evidence.items():
		opm_node = data.diagnoser.net.get_opm_node(fqn)
		if opm_node is not None:
			index = evidence.index(1.)
			status = opm_node.states[index]
		elif evidence == [0.0, 1.0]:
			status = '❌'
		elif evidence == [1.0, 0.0]:
			status = '✔️'
		else:
			status = '[' + ', '.join(f'{e:.2f}' for e in evidence) + ']'
		rows.append({'result': fqn, 'status': status})

	with ui.scroll_area().classes('w-full'):
		ui.table(columns=columns, rows=rows)
	ui.button('Download', on_click=lambda: ui.download(export(data), f'results_{data.cluster.name}.xlsx'))
