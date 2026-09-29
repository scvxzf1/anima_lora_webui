def test_parser_schema_preserves_manual_integer_contract():
    import train
    from library.config import schema

    parser = train.setup_parser()
    schema.populate_schema(parser)
    args = parser.parse_args(["--auto_block_swap", "--blocks_to_swap", "12"])
    assert args.auto_block_swap and args.blocks_to_swap == 12
    assert schema.CONFIG_SCHEMA["auto_block_swap"].type == "bool"
    assert schema.CONFIG_SCHEMA["blocks_to_swap"].type == "int"
    assert schema.CONFIG_SCHEMA["auto_block_swap_max_trials"].type == "int"
    assert schema.CONFIG_SCHEMA["auto_block_swap_interval"].type == "int"
    assert schema.CONFIG_SCHEMA["auto_block_swap_mode"].type == "str"
    assert schema.CONFIG_SCHEMA["auto_block_swap_vram_reserve_percent"].type == "float"
    assert schema.CONFIG_SCHEMA["auto_block_swap_preference"].type == "str"
