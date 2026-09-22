import pytest
import torch

from bench.adaptive_runtime.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint


def scaled_update(model, optimizer, scaler):
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.randn(3, 4)).square().mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()


def fixture(tmp_path):
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", init_scale=128, growth_interval=2)
    initial = [p.detach().clone() for p in model.parameters()]
    scaled_update(model, optimizer, scaler)
    signature = {"loss_scale": 128}
    path = tmp_path / "scaled.pt"
    save_checkpoint(path, network=model, optimizer=optimizer, initial=initial,
                    updates=[{"step": 1}], signature=signature, device="cpu", scaler=scaler)
    return path, model, optimizer, scaler, signature


def test_scaled_checkpoint_restores_next_update_and_growth_tracker(tmp_path):
    with torch.random.fork_rng(devices=[]):
        path, model, optimizer, scaler, signature = fixture(tmp_path)
        payload = read_checkpoint(path, signature=signature)
        assert payload["schema"] == "adaptive_fixed_input_probe_v2"
        assert payload["scaler"]["_growth_tracker"] == 1
        scaled_update(model, optimizer, scaler)
        assert scaler.get_scale() == 256
        expected_rng = torch.get_rng_state()
        restored = torch.nn.Linear(4, 2)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_scaler = torch.amp.GradScaler("cpu", init_scale=2, growth_interval=99)
        restore_checkpoint(payload, network=restored, optimizer=restored_optimizer,
                           params=list(restored.parameters()), device="cpu", scaler=restored_scaler)
        scaled_update(restored, restored_optimizer, restored_scaler)
        torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(restored_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
        assert restored_scaler.state_dict() == scaler.state_dict()
        assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("field,value", [("scale", float("nan")), ("scale", 0),
                                         ("growth_factor", 1), ("backoff_factor", 1),
                                         ("growth_interval", 0), ("_growth_tracker", -1),
                                         ("_growth_tracker", 2)])
def test_invalid_scaler_state_is_rejected(tmp_path, field, value):
    path, _, _, _, signature = fixture(tmp_path)
    payload = torch.load(path, weights_only=True)
    payload["scaler"][field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match="scaler"):
        read_checkpoint(path, signature=signature)


def test_legacy_scaled_checkpoint_and_missing_destination_scaler_rejected(tmp_path):
    path, model, optimizer, _, signature = fixture(tmp_path)
    payload = read_checkpoint(path, signature=signature)
    with pytest.raises(ValueError, match="destination scaler"):
        restore_checkpoint(payload, network=model, optimizer=optimizer,
                           params=list(model.parameters()), device="cpu")
    payload.pop("scaler")
    payload["schema"] = "adaptive_fixed_input_probe_v1"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Legacy"):
        read_checkpoint(path, signature=signature)


def test_cannot_save_scaled_contract_without_scaler(tmp_path):
    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError, match="requires scaler"):
        save_checkpoint(tmp_path / "bad.pt", network=model,
                        optimizer=torch.optim.AdamW(model.parameters()), initial=[], updates=[],
                        signature={"loss_scale": 128}, device="cpu")
    assert not (tmp_path / "bad.pt").exists()
