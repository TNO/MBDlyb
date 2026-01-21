# -*- coding: utf-8 -*-
"""
	Copyright (c) 2023 - 2026 TNO-ESI
	All rights reserved.
"""
from typing import Type, Optional, Union

from mbdlyb.base import MBDNet, MBDReasoner
from .bayesnet import BayesNet, BayesNetReasoner
from .tensor_network import TensorNet, TensorNetReasoner
from .markovnet import MarkovNet, MarkovNetReasoner


def select_reasoner(net: MBDNet, reasoner_klass: Optional[Type[MBDReasoner]] = None) -> Union[BayesNetReasoner, TensorNetReasoner, MarkovNetReasoner]:
	if reasoner_klass is None:
		if isinstance(net, TensorNet):
			reasoner = TensorNetReasoner(net)
		elif isinstance(net, MarkovNet):
			reasoner = MarkovNetReasoner(net)
		elif isinstance(net, BayesNet):
			reasoner = BayesNetReasoner(net)
		else:
			raise RuntimeError(f'No suitable reasoner found for \'{net.__class__} {net.name}')
	else:
		reasoner = reasoner_klass(net)
	return reasoner
