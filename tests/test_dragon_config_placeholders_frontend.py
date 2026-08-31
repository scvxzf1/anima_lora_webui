from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.frontend_test_support import REPO_ROOT, STATIC_DIR, node_syntax_check


METHOD_PLACEHOLDER_KEYS = {
    "network_args",
    "alpha_rank_scale",
    "layer_start",
    "balance_loss_weight",
    "balance_loss_warmup_ratio",
    "network_router_lr_scale",
    "router_targets",
    "sigma_feature_dim",
    "per_bucket_balance_weight",
    "sigma_bucket_boundaries",
    "ip_image_drop_p",
    "easycontrol_drop_p",
    "easycontrol_cond_noise_max",
    "fei_feature_dim",
    "fei_sigma_low_div",
    "router_hidden_dim",
    "router_tau",
    "fera_fecl_weight",
    "fera_num_bands",
}


def test_method_placeholders_show_a_value_and_its_effect() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon config placeholder checks")

    module_uri = (
        STATIC_DIR / "js/config/catalog/field-placeholders.js"
    ).resolve().as_uri()
    script = f"""
const mod = await import({json.dumps(module_uri + '?placeholder-test')});
const selected = Object.fromEntries(
  {json.dumps(sorted(METHOD_PLACEHOLDER_KEYS))}.map((key) => [key, mod.FIELD_PLACEHOLDER_ZH[key]])
);
console.log(JSON.stringify({{
  selected,
  fallback: mod.configFieldPlaceholder('unknown_key', '未知参数'),
  prototypeFallback: mod.configFieldPlaceholder('toString', '原型同名参数'),
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert set(payload["selected"]) == METHOD_PLACEHOLDER_KEYS
    assert all(payload["selected"].values())
    assert all(
        placeholder.startswith("例如：") and "，" in placeholder
        for placeholder in payload["selected"].values()
    )
    assert payload["selected"]["alpha_rank_scale"] == "例如：1.0，秩按时间步线性变化"
    assert payload["selected"]["easycontrol_cond_noise_max"].startswith("例如：0.1，")
    assert payload["selected"]["router_tau"].endswith("越低专家选择越集中")
    assert payload["fallback"] == "例如：未知参数…"
    assert payload["prototypeFallback"] == "例如：原型同名参数…"


def test_config_page_uses_curated_placeholders_only_for_text_controls() -> None:
    page = (STATIC_DIR / "js/dragon-ui/pages/config-page.js").read_text(encoding="utf-8")

    assert "configFieldPlaceholder(key, label)" in page
    assert "const placeholder = `例如：${label}…`;" not in page
    assert page.count('placeholder="${escapeHtml(placeholder)}"') == 2

    select_branch = page.split("if (options && !booleanField)", 1)[1].split(
        "} else if (booleanField)", 1
    )[0]
    toggle_branch = page.split("} else if (booleanField)", 1)[1].split(
        "} else if (key.includes('prompt')", 1
    )[0]
    assert "placeholder=" not in select_branch
    assert "placeholder=" not in toggle_branch


@pytest.mark.parametrize(
    "relative_module",
    [
        "js/config/catalog/field-placeholders.js",
        "js/dragon-ui/pages/config-page.js",
    ],
)
def test_config_placeholder_modules_parse(relative_module: str) -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon config placeholder checks")
    result = node_syntax_check(relative_module)
    assert result.returncode == 0, result.stderr
