from pathlib import Path

import pytest

from tests.web_config_test_support import _patch_config_service_paths, _write_minimal_config_tree
from web.services import config_service
from web.services.config import raw_files


@pytest.fixture
def config_tree(tmp_path, monkeypatch):
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    return configs


def test_rename_preserves_content_group_and_order(config_tree):
    source = 'configs/imported/lora.toml'
    target = 'configs/imported/renamed.toml'
    original = (config_tree / 'imported/lora.toml').read_bytes()
    ok, message, group = config_service.create_config_file_group('Test', kind='training')
    assert ok, message
    ok, message, _ = config_service.move_config_file_to_group(source, group['id'])
    assert ok, message
    specs = config_service._load_config_file_group_specs()
    for spec in specs:
        if spec['id'] == group['id']:
            spec['order'] = [source]
    config_service._save_config_file_group_specs(specs)

    ok, message = config_service.rename_raw_file('imported/lora.toml', target)
    assert ok, message
    assert not (config_tree / 'imported/lora.toml').exists()
    assert (config_tree / 'imported/renamed.toml').read_bytes() == original
    specs = config_service._load_config_file_group_specs()
    moved = next(spec for spec in specs if spec['id'] == group['id'])
    assert target in moved['files']
    assert moved['order'] == [target]
    assert all(source not in spec[key] for spec in specs for key in ('files', 'order', 'exclude'))


@pytest.mark.parametrize('target', [
    'configs/imported/../imported/new.toml', '/configs/imported/new.toml',
    'configs/datasets/new.toml', 'configs/imported/new.txt', '',
])
def test_rename_rejects_invalid_target_without_touching_source(config_tree, target):
    source = config_tree / 'imported/lora.toml'
    original = source.read_bytes()
    ok, _ = config_service.rename_raw_file('configs/imported/lora.toml', target)
    assert not ok
    assert source.read_bytes() == original


def test_rename_rejects_collision_and_locks(config_tree):
    source = 'configs/imported/lora.toml'
    target = config_tree / 'imported/existing.toml'
    target.write_text('output_name = "keep"\n')
    ok, _ = config_service.rename_raw_file(source, 'configs/imported/existing.toml')
    assert not ok
    assert target.read_text() == 'output_name = "keep"\n'
    config_service.set_user_file_lock(source, True)
    ok, _ = config_service.rename_raw_file(source, 'configs/imported/new.toml')
    assert not ok
    assert (config_tree / 'imported/lora.toml').exists()
    assert not (config_tree / 'imported/new.toml').exists()


def test_rename_target_created_during_operation_is_not_overwritten(config_tree, monkeypatch):
    original_link = Path.hardlink_to

    def competing_create(target, source):
        target.write_text('output_name = "concurrent"\n')
        return original_link(target, source)

    monkeypatch.setattr(Path, 'hardlink_to', competing_create)
    ok, _ = config_service.rename_raw_file('configs/imported/lora.toml', 'configs/imported/new.toml')
    assert not ok
    assert (config_tree / 'imported/lora.toml').exists()
    assert (config_tree / 'imported/new.toml').read_text() == 'output_name = "concurrent"\n'


def test_rename_metadata_failure_restores_source(config_tree, monkeypatch):
    config_service._save_config_file_group_specs([{
        'id': 'imported', 'label': 'Imported', 'trainable': True,
        'files': ['configs/imported/lora.toml'],
    }])
    original = raw_files._call_file_groups_impl
    failed = False

    def fail_once(name, *args):
        nonlocal failed
        if name == '_save_config_file_group_specs' and not failed:
            failed = True
            raise OSError('injected failure')
        return original(name, *args)

    monkeypatch.setattr(raw_files, '_call_file_groups_impl', fail_once)
    ok, _ = config_service.rename_raw_file('configs/imported/lora.toml', 'configs/imported/new.toml')
    assert not ok
    assert (config_tree / 'imported/lora.toml').exists()
    assert not (config_tree / 'imported/new.toml').exists()
    assert config_service._load_config_file_group_specs()[0]['files'] == ['configs/imported/lora.toml']
