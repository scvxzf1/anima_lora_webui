import pytest
import torch

from bench.adaptive_runtime.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(3, 4)).square().mean().backward()
    optimizer.step()


def test_complete_checkpoint_restores_optimizer_rng_and_next_update(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260921)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        initial = [p.detach().clone() for p in model.parameters()]
        update(model, optimizer)
        signature = {"inputs_sha256": "fixture", "torch": str(torch.__version__)}
        path = tmp_path / "step-1.pt"
        save_checkpoint(path, network=model, optimizer=optimizer, initial=initial,
                        updates=[{"step": 1}], signature=signature, device="cpu")
        payload = read_checkpoint(path, signature=signature)
        update(model, optimizer)
        expected_rng = torch.get_rng_state()
        restored = torch.nn.Linear(4, 2)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_initial, updates = restore_checkpoint(
            payload, network=restored, optimizer=restored_optimizer,
            params=list(restored.parameters()), device="cpu",
        )
        update(restored, restored_optimizer)
        torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(restored_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert updates == [{"step": 1}]
        torch.testing.assert_close(restored_initial, initial, rtol=0, atol=0)
        with pytest.raises(ValueError, match="contract"):
            read_checkpoint(path, signature={"inputs_sha256": "different"})


def test_failed_checkpoint_write_never_publishes_partial_state(tmp_path, monkeypatch):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters())

    def failure(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(torch, "save", failure)
    with pytest.raises(OSError, match="disk failure"):
        save_checkpoint(tmp_path / "step.pt", network=model, optimizer=optimizer,
                        initial=[], updates=[{"step": 1}], signature={}, device="cpu")
    assert not list(tmp_path.iterdir())


def test_incomplete_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "broken.pt"
    torch.save({"network": {}}, path)
    with pytest.raises(ValueError, match="Incomplete"):
        read_checkpoint(path, signature={})


@pytest.mark.parametrize("corrupt", ["optimizer", "initial"])
def test_nonfinite_resume_state_is_rejected(tmp_path, corrupt):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    initial = [p.detach().clone() for p in model.parameters()]
    update(model, optimizer)
    path = tmp_path / "step.pt"
    save_checkpoint(path, network=model, optimizer=optimizer, initial=initial,
                    updates=[{"step": 1}], signature={}, device="cpu")
    payload = torch.load(path, weights_only=True)
    if corrupt == "optimizer":
        next(iter(payload["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
    else:
        payload["initial"][0].fill_(float("inf"))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="optimizer|initial"):
        read_checkpoint(path, signature={})
