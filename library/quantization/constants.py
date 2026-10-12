"""Serialized NF4 state names, without an optional bitsandbytes import."""

STATE_KEYS = frozenset({
    'absmax', 'quant_map', 'nested_absmax', 'nested_quant_map',
    'quant_state.bitsandbytes__nf4',
})
