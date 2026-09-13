"""Inventory-only startup: all performance decisions happen in the live run."""

from .config import probe_arguments
from .process import run_process, write_result


def initialize(args, directory, *, runner=run_process):
    inventory_dir = directory / "inventory"
    inventory = runner(
        probe_arguments(args, inventory_dir, blocks=1, inventory=True),
        inventory_dir,
        timeout=args.auto_block_swap_timeout,
    )
    if inventory.get("status") != "inventory":
        raise RuntimeError(f"Dynamic AUTO inventory failed: {inventory.get('status')}")
    maximum = min(inventory["model_swap_limit"], inventory["host_swap_limit"])
    if maximum < 1:
        raise RuntimeError("Dynamic AUTO requires RAM for at least one swapped block")
    args._auto_swap_maximum = maximum
    args._auto_swap_block_bytes = inventory["block_bytes"]
    args._auto_swap_resolved = True
    args.blocks_to_swap = maximum
    args._auto_swap_report = str(directory / "summary.json")
    write_result(
        directory / "summary.json",
        {
            "status": "dynamic",
            "initial_blocks": maximum,
            "inventory": inventory,
            "seed": args.seed,
        },
    )
