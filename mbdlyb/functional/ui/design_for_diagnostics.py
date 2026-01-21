# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
import io
import re
from nicegui import ui, APIRouter, run, app
from zipfile import ZipFile, ZIP_DEFLATED

from mbdlyb.functional import Analyzer, DPTreeNode
from mbdlyb.functional.gdb import Cluster
from mbdlyb.ui.utils import Status, show_name, status

from .base import header, footer, goto, reasoner_class_dict

router = APIRouter(prefix='/cluster/{cluster_id}')


class UIDesignForDiagnosticsData:
	compute_button: ui.button = None
	export_button: ui.button = None
	_cluster_id: str = None
	cluster: Cluster = None
	analyzer: Analyzer = None
	costs_text: str = ''
	costs_visible: bool = False
	dp_to_export: list[str] = []
	dp_svg: dict[str, str] = {}
	results_available: bool = False

	def __init__(self, cluster_id: str):
		self._cluster_id = cluster_id

	def reset(self):
		self.cluster = Cluster.load(self._cluster_id)
		reasoner_klass_name = app.storage.general.get('reasoner')
		reasoner_klass = reasoner_class_dict[reasoner_klass_name]
		self.analyzer = Analyzer(self.cluster, reasoner_klass)
		self.results_available = False
		show_name.refresh(self.cluster.name)

	async def compute(self, event=None):
		status.refresh(Status.COMPUTING)
		if event is not None:
			event.sender.disable()

		self.health_results = await run.io_bound(self.analyzer.analyze)
		self.results_available = True
		operating_modes.refresh()
		signatures.refresh()
		diagnostic_procedures.refresh()
		overview.refresh()

		if event is not None:
			event.sender.enable()
		status.refresh(Status.READY)

		# Compute should only be pressed once
		self.compute_button.disable()

		# Export is enabled after compute finishes
		self.export_button.enable()


@router.page('/design_for_diagnostics/')
def design_for_diagnostics_main(cluster_id: str):
	editor_url = f'/cluster/{cluster_id}/'
	data: UIDesignForDiagnosticsData = UIDesignForDiagnosticsData(cluster_id)
	data.reset()

	header('Design for diagnostics')
	footer()

	with ui.row().classes('w-full gap-5 items-center'):
		show_name(data.cluster.name)
		status(Status.COMPUTING)
		data.compute_button = ui.button('Compute', icon='refresh', color='primary').on_click(data.compute)
		data.export_button = ui.button('Export', on_click=lambda: ui.download(data.analyzer.export(), f'design_for_diagnostics_{data.cluster.name}.xlsx'))
		data.export_button.disable()
		ui.space()
		ui.button(app.storage.general['mode'], icon='home', color='primary').on_click(lambda: goto(editor_url))

	with ui.grid(columns='1fr').classes('gap-1 min-w-[600px]'):
		with ui.column().classes('border rounded-borders p-2 h-full'):
			with ui.row().classes('w-full items-center'):
				ui.label('Operating modes').classes('text-h5')
			operating_modes(data)
	with ui.grid(columns='1fr').classes('gap-1 min-w-[600px]'):
		with ui.column().classes('border rounded-borders p-2 h-full'):
			with ui.row().classes('w-full items-center'):
				ui.label('Signatures').classes('text-h5')
				ui.space()
				filter = ui.input('Filter')
			signatures(data, filter)
	with ui.grid(columns='1fr').classes('gap-1 min-w-[600px]'):
		with ui.column().classes('border rounded-borders p-2 h-full'):
			with ui.row().classes('w-full items-center'):
				ui.label('Diagnostic procedures').classes('text-h5')
				ui.space()
				button = ui.button('Export diagrams')
			diagnostic_procedures(data, button)
	with ui.grid(columns='1fr').classes('gap-1 min-w-[600px]'):
		with ui.column().classes('border rounded-borders p-2 h-full'):
			with ui.row().classes('w-full items-center'):
				ui.label('Overview').classes('text-h5')
			overview(data)
	status.refresh(Status.READY)


@ui.refreshable
def operating_modes(data: UIDesignForDiagnosticsData):
	async def _save(dt, dt_ev, exp):
		data.analyzer.add_operating_mode(dt_ev)
		exp.close()
		exp.delete()

	with ui.scroll_area().classes('w-full'):
		if data.results_available:
			columns = [{'name': 'opm', 'label': 'Operating mode selector', 'field': 'opm', 'align': 'left'}, 
			           {'name': 'val', 'label': 'Selected mode', 'field': 'val', 'align': 'left'}]
			rows = [{'opm': opm, 'val': data.analyzer.selected_operating_modes[opm]} for opm in sorted(data.analyzer.selected_operating_modes)]
			unselected_opm = [opm for opm in data.cluster.get_all_opm_nodes().values() if opm.fqn not in data.analyzer.selected_operating_modes]
			rows += [{'opm': opm.fqn, 'val': ' | '.join(sorted(opm.states))} for opm in sorted(unselected_opm, key=lambda opm: opm.fqn)]
			ui.table(columns=columns, rows=rows, row_key='opm')
		else:
			for opm in data.cluster.get_all_opm_nodes().values():
				ev = dict()
				with ui.expansion(opm.fqn).classes('w-full') as exp:
					ui.select(opm.states, label='Operating mode').bind_value(ev, opm.fqn).classes('w-64')
					ui.button('Save', on_click=lambda t=opm, evd=ev, e=exp: _save(t, evd, e))


@ui.refreshable
def signatures(data: UIDesignForDiagnosticsData, filter: ui.input):
	if data.analyzer is None or data.analyzer.signatures is None:
		ui.label('No results to show yet.').classes('text-grey')
		return
	
	signatures = data.analyzer.signatures.reset_index(level=['Node', 'State'])
	signatures['Node'] = signatures['Node'].apply(str)
	signatures.columns = list(map(str, signatures.columns))
	with ui.column().classes('p-2'):
		table = ui.table.from_pandas(signatures, pagination=0)
		table.bind_filter(filter, 'value')
	for col in table.columns:
		col['align'] = 'left'
		col['sortable'] = True

	# first render the entire table, fix the width of the first column and then enable pagination
	async def fix_table_width():
		await ui.run_javascript(f'''
			el = document.querySelector("#c{str(table.id)} th");
			el.width = el.offsetWidth;
		''')
		table.pagination = {'rowsPerPage': 10, 'sortBy': 'Node'}
		button.set_visibility(False)
	button = ui.button('Fix width', on_click=fix_table_width)
	button.run_method('click')


class DiagnosticProcedureDiagram:
	_node_cnt: int
	_diagram: list[str]

	def __init__(self, data: UIDesignForDiagnosticsData, dp_name: str):
		root_node = data.analyzer.get_diagnostic_procedure_tree(dp_name)

		# start drawing at the root-node
		self._node_cnt = 0
		self._diagram = ['flowchart']
		self._diagram.append(f'subgraph "<b>{dp_name}</b>"')
		self._diagram.append('direction TB')
		self._draw_node(root_node)

		# close diagnostic procedure
		self._diagram.append('end')

		# add color mapping
		for name, color in DPTreeNode.get_color_mapping().items():
			self._diagram.append(f'classDef {name} fill:{color};')

	def _draw_node(self, node: DPTreeNode, prev_node: int = 0, edge_label: str = ''):

		self._node_cnt += 1
		new_node = self._node_cnt
		if node.type == 'Hardware' or node.type == 'NonDiagnosableHW':
			hw_names = '<br/>-----<br/>'.join(node.name.split('\n'))
			self._diagram.append(f'{new_node}[[{hw_names}]]')
		else:
			self._diagram.append(f'{new_node}{{{{{node.name}}}}}')
		if node.type == 'NonDiagnosableHW':
			self._diagram.append(f'class {new_node} NonDiagnosableHW')
		else:
			self._diagram.append(f'class {new_node} {node.type}')
		self._diagram.append(f'click {new_node} call emitEvent("mermaid_click", "{node.name}", "{node.cost}")')
		if edge_label:
			self._diagram.append(f'{prev_node} --> |{edge_label}| {new_node}')

		for edge in node.edges:
			edge_label = '<br/>'.join([f'{result_name} = {value}' for result_name, value in edge.results])
			self._draw_node(edge.target, new_node, edge_label)

	@property
	def diagram(self) -> str:
		return '\n'.join(self._diagram)


@ui.refreshable
def diagnostic_procedures(data: UIDesignForDiagnosticsData, export_button: ui.button):
	if data.analyzer is None or data.analyzer.dps is None:
		ui.label('No results to show yet.').classes('text-grey')
		return

	def update_costs_label(name: str, costs: str):
		name = ', '.join(name.split('\n'))
		data.costs_text = f'Costs for <b>{name}</b>: {costs}'
		data.costs_visible = True

	def hide_costs_label():
		data.costs_visible = False

	dp_names = sorted(data.analyzer.dps.keys())
	dp_tabs = dict()
	with ui.tabs().classes('px-2') as tabs:
		for dp_name in dp_names:
			dp_tabs[dp_name] = ui.tab(dp_name)
	tabs.on_value_change(hide_costs_label)
	mermaid_ids = {}
	with ui.tab_panels(tabs) as panels:
		for dp_name in dp_names:
			with ui.tab_panel(dp_tabs[dp_name]).classes('p-2') as tab_panel:
				panels.value = panels.value or tab_panel
				dp = data.analyzer.dps[dp_name].reset_index(level=['Node', 'State'])
				dp['Node'] = dp['Node'].apply(str)
				dp.columns = list(map(str, dp.columns))				
				table = ui.table.from_pandas(dp, pagination={'rowsPerPage': 0, 'sortBy': 'Node'})
				for col in table.columns:
					col['align'] = 'left'
					col['sortable'] = True
				dpd = DiagnosticProcedureDiagram(data, dp_name)
				mermaid_config = {
					'securityLevel': 'loose',
					'theme': 'neutral',
					'themeVariables': {
						'fontSize': '14px'
					}
				}
				mm = ui.mermaid(dpd.diagram, config=mermaid_config).classes('w-full')
				mermaid_ids[dp_name] = mm.id
	ui.html(sanitize=False).bind_content_from(data, 'costs_text').bind_visibility_from(data, 'costs_visible')
	ui.on('mermaid_click', lambda event: update_costs_label(*event.args))
	
	def export_start():
		data.dp_to_export = dp_names
		data.dp_svg = {}
		export_store_svg()
	export_button.on_click(export_start)

	def export_store_svg(svg={}):
		data.dp_svg |= svg
		if data.dp_to_export:
			dp = data.dp_to_export.pop()
			tabs.set_value(dp)
			export_switch_tab()
		else:
			ui.download(
				export_zip_svg(data.dp_svg, data.cluster.name),
				f'diagnostic_procedures_{data.cluster.name}.zip')

	def export_switch_tab():
		dp = tabs.value._props['name'] if isinstance(tabs.value, ui.tab_panel) else tabs.value			
		id = mermaid_ids[dp]
		ui.run_javascript(f'''
			svg = document.querySelector("#c" + {id} + " > svg");
			emitEvent('mermaid_export', {{
				{dp}: svg.outerHTML,
			}});
		''')
	ui.on('mermaid_export', lambda evt: export_store_svg(evt.args))

	def export_zip_svg(all_svg: dict[str,str], cluster_name: str) -> bytes:
		bytes = io.BytesIO()
		z = ZipFile(bytes, 'w', compression=ZIP_DEFLATED, compresslevel=5)
		for dp_name, svg in all_svg.items():
			svg = svg.replace(r'<br>', r'<br/>')
			svg = re.sub(r'id="<b>(.*)</b>"', r'id="\1"', svg)
			z.writestr(f'{cluster_name}-{dp_name}.svg', svg)
		z.close()
		bytes.seek(0,0)
		return bytes.read()


@ui.refreshable
def overview(data: UIDesignForDiagnosticsData):
	if data.analyzer is None or data.analyzer.overview is None:
		ui.label('No results to show yet.').classes('text-grey')
		return

	overview = data.analyzer.overview.rename_axis('DP').reset_index()
	with ui.column().classes('p-2'):
		table = ui.table.from_pandas(overview, pagination={'rowsPerPage': 0, 'sortBy': 'DP'})
	for col in table.columns:
		col['align'] = 'left'
		col['sortable'] = True
