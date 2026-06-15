"""Dynamo compile-budget helpers.

PyTorch Dynamo 的部分配置是 ContextVar 覆盖：主线程里普通设置
``torch._dynamo.config.recompile_limit = N``，到 AOTAutograd / backward compile
上下文里可能又读回默认值。训练里这会表现为明明提前放大了预算，
checkpoint 重算阶段仍然在默认 8 次限制下触发 recompile storm。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def pin_dynamo_limit(name: str, value: int) -> int:
    """Raise a Dynamo recompile budget across all compile contexts.

    同时设置当前 ContextVar 覆盖和底层 canonical config entry 的 ``default``。
    后者是 backward/AOTAutograd compile context 读取到的 fallback，避免预算
    在 checkpoint recompute 侧退回默认 8。
    """

    import torch._dynamo as _dynamo

    cfg_mod = _dynamo.config
    target = max(getattr(cfg_mod, name), int(value))
    setattr(cfg_mod, name, target)
    try:
        entry = cfg_mod._config[name]
        canon = (entry.alias or name).rsplit(".", 1)[-1]
        cfg_mod._config[canon].default = target
    except Exception as exc:  # noqa: BLE001 - torch internals vary by version
        logger.warning(
            "could not pin dynamo %s default (%s); budget may revert in "
            "backward compile context",
            name,
            exc,
        )
    return target
