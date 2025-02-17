# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2025 TNO-ESI
	All rights reserved.
"""
from argparse import ArgumentParser

from neomodel import config
from nicegui import app, ui

from .base import page
from .clusters import router as cluster_router
from .functions import router as function_router
from .hardware import router as hardware_router
from .observables import router as observable_router
from .diagnostic_tests import router as diagnostic_test_router
from .operating_modes import router as opm_router
from .diagnostic_test_results import router as diagnostic_test_result_router
from .diagnoser import router as diagnoser_router
from .design_for_diagnostics import router as design_for_diagnostics_router


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
	app.storage.general['show_diagrams'] = True
	app.storage.general['auto_compute'] = False

	page('Select a cluster to start editing...')
	ui.link('...or create one!', target='/cluster/new/')
	ui.link('You can also import a new cluster.', target='/cluster/import/')

	app.include_router(cluster_router)
	app.include_router(function_router)
	app.include_router(hardware_router)
	app.include_router(observable_router)
	app.include_router(diagnostic_test_router)
	app.include_router(opm_router)
	app.include_router(diagnostic_test_result_router)
	app.include_router(diagnoser_router)
	app.include_router(design_for_diagnostics_router)

	ui.run(title='Model Editor', show=False, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
	run()
