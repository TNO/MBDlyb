# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
import asyncio
import inspect
from dataclasses import dataclass
from typing import Optional, Callable, Iterable, Union, Any

from nicegui import app, ui

from mbdlyb.formalisms import BayesNetReasoner, TensorNetReasoner, MarkovNetReasoner
from mbdlyb.functional.gdb import MBDElement, Cluster, FunctionalNode
from mbdlyb.ui.helpers import goto


reasoner_class_dict = {'BayesNetReasoner': BayesNetReasoner,
					   'TensorNetReasoner': TensorNetReasoner,
					   'MarkovNetReasoner': MarkovNetReasoner}


@dataclass
class Button:
	"""
	Class for keeping track of button options
	"""
	label: Optional[str | Callable[[MBDElement], str]] = None
	icon: Optional[str | Callable[[MBDElement], str]] = None
	color: Optional[str | Callable[[MBDElement], str]] = None
	handler: Optional[str | Callable[[], any] | Callable[[MBDElement], any]] = None
	tooltip: Optional[str | Callable[[MBDElement], str]] = None

	def show(self, obj: MBDElement = None, hide_label=False, btn_props: str = None, tooltip_classes: str = None):
		with ui.button(text=None if hide_label else self._eval(self.label, obj), icon=self._eval(self.icon, obj),
					   color=self._eval(self.color, obj), on_click=self._eval_handler(self.handler, obj)).props(
				btn_props or ''):
			if self.tooltip:
				ui.tooltip(self._eval(self.tooltip)).classes(tooltip_classes or '')

	def show_icon(self, obj: MBDElement = None, btn_props: str = None, tooltip_classes: str = None):
		with ui.icon(self._eval(self.icon, obj), color=self._eval(self.color, obj)).on('click', self._eval_handler(
				self.handler, obj)).classes('cursor-pointer').props(btn_props or ''):
			if self.tooltip:
				ui.tooltip(self._eval(self.tooltip, obj)).classes(tooltip_classes or '')

	@staticmethod
	def _eval(attr: Optional[str | Callable[[MBDElement], str]], obj: Optional[MBDElement] = None) -> Optional[str]:
		if attr is None:
			return None
		if isinstance(attr, str):
			return attr.format(obj.name if obj is not None else None)
		elif isinstance(attr, Callable):
			return attr(obj)
		return attr

	@staticmethod
	def _eval_handler(attr: Optional[str | Callable[[], any] | Callable[[MBDElement], any]], obj: Optional[MBDElement] = None) -> Optional[Callable]:
		if attr is None:
			return None
		if isinstance(attr, str):
			return lambda: goto(attr)
		elif isinstance(attr, Callable):
			if obj is None:
				return attr
			return lambda: attr(obj)
		return attr


@dataclass
class ConditionalButton(Button):
	condition: Optional[Callable[[MBDElement], bool]] = None

	def show(self, obj: MBDElement = None, hide_label=False, btn_props: str = None, tooltip_classes: str = None):
		if self.condition is None or self.condition(obj):
			super().show(obj, hide_label, btn_props, tooltip_classes)

	def show_icon(self, obj: MBDElement = None, btn_props: str = None, tooltip_classes: str = None):
		if self.condition is None or self.condition(obj):
			super().show_icon(obj, btn_props, tooltip_classes)


def build_menu_tree(parent: Cluster = None):
	return [
		{'id': cluster.uid, 'label': cluster.name, 'children': build_menu_tree(cluster)} for cluster in
		(Cluster.nodes.has(net=False).order_by('name') if parent is None else parent.subnets.order_by('name'))
	]


def _buttons(buttons: list[list | Button]):
	for button in buttons:
		if isinstance(button, Button):
			button.show(btn_props='outline', tooltip_classes='text-xs')
		elif isinstance(button, list) and len(button) >= 2:
			with ui.button_group():
				_buttons(button)
		else:
			import warnings
			warnings.warn('Could not add buttons to page.')


def _settings(dialog: ui.dialog):
	def emit_show_diagram():
		ui.run_javascript('''emitEvent('show_diagram', '');''')

	with dialog, ui.card():
		ui.label('Settings').classes('text-h5')
		ui.switch('Model editor - show diagrams').bind_value(app.storage.general, 'show_diagrams').on_value_change(emit_show_diagram)
		ui.switch('Functional diagnoser - auto-compute').bind_value(app.storage.general, 'auto_compute')

		# Since the real class objects are not serializable, we store the reasoner class names:
		ui.select({'BayesNetReasoner': 'BayesNet', 'TensorNetReasoner': 'TensorNet', 'MarkovNetReasoner': 'MarkovNet'},
				  label='Reasoner').bind_value(app.storage.general, 'reasoner')


def header(text: str, url: str = '/'):
	ui.page_title(text)
	with ui.header(elevated=True).style('background-color: #3874c8'):
		with ui.link(target=url).classes(remove='nicegui-link'):
			ui.label(text).classes('text-h4')
		ui.space()
		dialog = ui.dialog()
		_settings(dialog)
		ui.icon('settings').classes('text-h4').on('click', dialog.open)


def footer():
	with ui.footer(elevated=False):
		ui.label(
			'This prototype tool has been developed by TNO-ESI. All rights reserved.').classes('fit text-right')


def page(title: str = None, cluster: Cluster = None, buttons=None):
	def _cluster_trace(c: Cluster) -> list[Cluster]:
		return (_cluster_trace(c.get_net()) if not c.at_root else []) + [c]

	cluster_trace = None if cluster is None else _cluster_trace(cluster)

	header('Model Editor')
	with ui.left_drawer(bottom_corner=True).style('background-color: #d7e3f4'):
		menu = ui.tree(build_menu_tree(), on_select=lambda c_id: goto(f'/cluster/{c_id.value}'))
		if cluster_trace is not None:
			menu.expand([c.uid for c in cluster_trace])
	footer()
	if cluster_trace:
		with ui.row():
			for c in cluster_trace:
				ui.link(c.name, target=f'/cluster/{c.uid}/')
		ui.separator()
	if title:
		with ui.row().classes('w-full'):
			ui.label(title).classes('text-h5')
			if buttons:
				ui.space()
				_buttons(buttons)


@dataclass
class TableColumn:
	header: str
	value: str | Callable[[MBDElement], str]
	url: Optional[Callable[[MBDElement], str]] = None
	tooltip: Optional[str | Callable[[MBDElement], str]] = None

	def show(self, obj: MBDElement, classes: Optional[str] = None, tooltip_classes: Optional[str] = None):
		if isinstance(self.value, str):
			label = obj.__getattribute__(self.value)
		elif isinstance(self.value, Callable):
			label = self.value(obj)
		else:
			return
		with ui.row().classes(classes):
			with (ui.link(label, self.url(obj)) if self.url else ui.label(label)):
				if self.tooltip:
					ui.tooltip(
						obj.__getattribute__(self.tooltip) if isinstance(self.tooltip, str) else self.tooltip(
							obj)).classes(tooltip_classes or '')


@dataclass
class TableMultiColumn(TableColumn):
	list_attribute: str = None
	order_by: Optional[str] = None
	separator: Optional[str] = '	|'

	def show(self, obj: MBDElement, classes: Optional[str] = None, tooltip_classes: Optional[str] = None):
		elements = obj.__getattribute__(self.list_attribute) if not (
					self.order_by or isinstance(self.value, str)) else obj.__getattribute__(
			self.list_attribute).order_by(self.order_by or self.value)
		if not elements:
			ui.label().classes(classes or '')
			return
		l_idx = len(elements) - 1
		with ui.row().classes(classes):
			for idx, o in enumerate(elements):
				super().show(o, 'inline')
				if idx < l_idx:
					ui.label(self.separator).classes('inline')


@dataclass
class Table:
	title: str
	rows: Iterable
	columns: list[TableColumn]
	obj: Optional[object] = None
	table_actions: Optional[list[Button]] = None
	row_actions: Optional[list[Button]] = None

	def show(self):
		with ui.row().classes('w-full'):
			ui.label(self.title).classes('text-h6')
			if self.table_actions:
				ui.space()
				for table_action in self.table_actions:
					table_action.show(self.obj, btn_props='outline', tooltip_classes='text-xs')

		def_classes = 'p-3 border border-gray-200'
		with ui.grid(columns=' '.join(['auto'] * len(self.columns)) + (' 1fr' if self.row_actions else '')).classes('w-full gap-0'):
			if not self.rows:
				ui.label('No entries found.')
				return
			for column in self.columns:
				ui.label(column.header).classes(f'font-bold {def_classes}')
			if self.row_actions:
				ui.label().classes(def_classes)
			for o in self.rows:
				for column in self.columns:
					column.show(o, def_classes)
				if self.row_actions:
					with ui.element('div').classes(f'w-full text-right {def_classes}'):
						for action in self.row_actions:
							action.show_icon(o)


# GENERIC DIALOGS

def confirm(title: str, message: str, confirm_text: str, on_confirm: Union[Callable, tuple[Callable, tuple[Any]]], confirm_color: str = 'primary'):
	def _on_cancel():
		dialog.close()
		dialog.clear()

	def _on_confirm():
		if isinstance(on_confirm, tuple):
			oc_fn, args = on_confirm
		else:
			oc_fn, args = on_confirm, ()
		_r = oc_fn(*args)
		if inspect.iscoroutine(_r):
			asyncio.create_task(_r)
		_on_cancel()

	with ui.dialog(value=True) as dialog, ui.card():
		ui.label(title).classes('text-h6')
		ui.label(message)
		with ui.row():
			ui.button('Cancel', color='secondary', on_click=_on_cancel).props('outline')
			ui.button(confirm_text, color=confirm_color, on_click=_on_confirm)
		dialog.on('close', dialog.clear)


def confirm_delete(obj, redirect_url: str):
	def _delete():
		if hasattr(obj, 'on_delete') and callable(obj.on_delete):
			obj.on_delete()
		obj.delete()
		goto(redirect_url)

	confirm('Delete confirmation', f'Are you sure you want to delete {type(obj).__name__} {obj}?', 'Yes', _delete,
			'negative')


def confirm_delete_relation(relation: str, start: MBDElement, end: MBDElement, redirect_url: str):
	def _delete():
		start.__getattribute__(relation).disconnect(end)
		goto(redirect_url)

	confirm('Delete confirmation',
			f'Are you sure you want to delete \'{relation}\' relation from  {start.fqn} to {end.fqn}?', 'Yes', _delete,
			'negative')
