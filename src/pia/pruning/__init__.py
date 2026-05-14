"""
Poda no estructurada, poda estructurada por ancho y flujo tipo lottery ticket.
"""

from pia.models.narrowable_chain_cnn import NarrowableChainCnn
from pia.pruning.lottery_ticket import iterative_magnitude_pruning
from pia.pruning.masks import WeightMaskRegistry
from pia.pruning.prune import (
    make_imp_param_selector,
    prune_globally_by_magnitude,
    select_conv_weight_params,
    select_imp_weight_params,
)
from pia.pruning.prune_structured import (
    chain_cnn_channel_l1_scores,
    count_parameters,
    narrow_chain_cnn_by_fraction,
)
from pia.pruning.resnet18_slim import (
    parameter_and_buffer_bytes,
    slim_resnet18_cifar_from_state_dict,
)
from pia.pruning.structured_ticket import iterative_structured_magnitude_pruning

__all__ = [
    "NarrowableChainCnn",
    "WeightMaskRegistry",
    "chain_cnn_channel_l1_scores",
    "count_parameters",
    "iterative_magnitude_pruning",
    "iterative_structured_magnitude_pruning",
    "make_imp_param_selector",
    "narrow_chain_cnn_by_fraction",
    "parameter_and_buffer_bytes",
    "prune_globally_by_magnitude",
    "select_conv_weight_params",
    "select_imp_weight_params",
    "slim_resnet18_cifar_from_state_dict",
]
