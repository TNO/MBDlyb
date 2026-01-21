# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
from argparse import ArgumentParser

from neomodel import config
from nicegui import app, ui

from .base import page
from .helpers import goto, Button
from .clusters import router as cluster_router
from .functions import router as function_router
from .hardware import router as hardware_router
from .observables import router as observable_router
from .diagnostic_tests import router as diagnostic_test_router
from .operating_modes import router as opm_router
from .diagnostic_test_results import router as diagnostic_test_result_router
from .diagnoser import router as diagnoser_router
from .design_for_diagnostics import router as design_for_diagnostics_router
from .validator import router as validator_router
from ..gdb import Cluster


@ui.page('/')
def landing_page():
	mode = app.storage.general.get('mode', 'Tree')
	import_btn = Button(None, 'upload_file', 'secondary', lambda: goto('/cluster/import/'), 'Import cluster')
	if mode == 'Tree':
		page('Clusters', hide_breadcrumbs=True, hide_menu_tree=True, buttons=[import_btn])
		with ui.list().props('separator').classes('w-full'):
			for cluster in Cluster.nodes.has(net=False).order_by('name'):
				with ui.item(on_click=lambda c=cluster: goto(f'/cluster/{c.uid}/')):
					with ui.item_section().props('avatar'):
						ui.icon('account_tree')
					with ui.item_section():
						ui.item_label(cluster.name)
	elif mode == 'Editor':
		page('Select a cluster on the left...', buttons=[
			Button(None, 'add', None, lambda: goto('/cluster/new/'), 'New cluster'),
			import_btn
		])
		ui.label('Select an existing cluster on the left, or use the buttons on the left to either create a new one or import one.')


def run():
	parser = ArgumentParser()
	parser.add_argument('--host', help='Hostname of your neo4j database.', required=False, default='localhost')
	parser.add_argument('--port', help='Port on which your neo4j database listens.', required=False, default=7687,
						type=int)
	parser.add_argument('--protocol', help='Protocol to use to connect to neo4j database.', required=False,
						default='bolt')
	parser.add_argument('--username', help='Username for neo4j authentication.', required=False, default='neo4j')
	parser.add_argument('--password', help='Password for neo4j authentication.', required=False, default='password')
	parser.add_argument('--database', help='Name of the neo4j database to use.', required=False, default=None)
	args = parser.parse_args()

	config.DATABASE_URL = f'{args.protocol}://{args.username}:{args.password}@{args.host}:{args.port}'
	if args.database is not None:
		config.DATABASE_URL += f'/{args.database}'

	# reload default settings every time the server is restarted
	default_settings = {
		'mode': 'Tree',
		'show_diagrams': True,
		'auto_compute': False,
		'reasoner': 'BayesNetReasoner'
	}
	for setting, default in default_settings.items():
		app.storage.general[setting] = app.storage.general.get(setting, default)

	app.include_router(cluster_router)
	app.include_router(function_router)
	app.include_router(hardware_router)
	app.include_router(observable_router)
	app.include_router(diagnostic_test_router)
	app.include_router(opm_router)
	app.include_router(diagnostic_test_result_router)
	app.include_router(diagnoser_router)
	app.include_router(design_for_diagnostics_router)
	app.include_router(validator_router)

	ui.run(title='Model Editor', show=False, reload=False, reconnect_timeout=600)


if __name__ in {"__main__", "__mp_main__"}:
	run()
