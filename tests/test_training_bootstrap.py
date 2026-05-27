from __future__ import annotations

import argparse
from types import SimpleNamespace

import train
from library.training.bootstrap import TrainingBootstrap


def test_trainer_uses_default_bootstrap():
    trainer = train.AnimaTrainer()

    assert isinstance(trainer.bootstrap, TrainingBootstrap)


def test_trainer_accepts_injected_bootstrap():
    class FakeBootstrap:
        def __init__(self) -> None:
            self.called = False

        def prepare_dataset(self, trainer, args):
            self.called = True
            return SimpleNamespace(
                train_dataset_group="train",
                val_dataset_group="val",
                current_epoch="epoch",
                current_step="step",
                collator="collator",
                use_user_config=True,
                use_dreambooth_method=False,
            )

    bootstrap = FakeBootstrap()
    trainer = train.AnimaTrainer(bootstrap=bootstrap)

    result = trainer._prepare_dataset(SimpleNamespace())

    assert bootstrap.called is True
    assert result == ("train", "val", "epoch", "step", "collator", True, False)


def test_bootstrap_batch_size_override_matches_trainer_wrapper():
    user_config = {
        "datasets": [
            {"batch_size": 1, "subsets": []},
            {"subsets": []},
        ]
    }

    train.AnimaTrainer._apply_train_batch_size_to_user_config(
        user_config,
        argparse.Namespace(train_batch_size=3),
    )

    assert [dataset["batch_size"] for dataset in user_config["datasets"]] == [3, 3]


def test_bootstrap_forwards_top_level_network_kwargs_with_cli_precedence():
    args = SimpleNamespace(
        network_args=["router_targets=cli_value"],
        router_targets="toml_value",
        use_lokr=True,
        gradient_accumulation_steps=4,
    )

    net_kwargs = TrainingBootstrap.build_net_kwargs(args)

    assert net_kwargs["router_targets"] == "cli_value"
    assert net_kwargs["use_lokr"] == "True"
    assert net_kwargs["gradient_accumulation_steps"] == "4"
