"""共享的可选 ONNX Runtime 推理实现。

本模块故意不在导入时加载 ``onnxruntime``。这样外部 API provider 和 WebUI
启动不依赖本地推理额外包；只有用户实际选择本地模型时才会检查运行时。
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from .base import LocalTaggingError, ProgressFn, TagResult

logger = logging.getLogger(__name__)


def onnxruntime_status() -> tuple[bool, str]:
    """检查可选依赖，不创建 session。"""

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except ImportError:
        return False, "未安装 onnxruntime；请安装 onnxruntime 或 onnxruntime-gpu"
    version = str(getattr(ort, "__version__", "")).strip()
    return True, f"onnxruntime {version}".strip()


@contextlib.contextmanager
def _silence_cuda_stderr() -> Iterator[None]:
    """抑制 CUDA EP 加载失败时 C 库直接写入 stderr 的噪声。"""

    try:
        sys.stderr.flush()
    except Exception:  # pragma: no cover - defensive around embedded stderr
        pass
    saved: int | None = None
    devnull = None
    try:
        saved = os.dup(2)
        devnull = open(os.devnull, "wb")
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        if saved is not None:
            os.dup2(saved, 2)
            os.close(saved)
        if devnull is not None:
            devnull.close()


class OnnxTaggerBase:
    """封装 session、batch、图像读取和 CUDA -> CPU fallback。"""

    name = "onnx"
    requires_service = False

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        *,
        overrides: dict[str, Any] | None = None,
    ):
        self.settings = dict(settings or {})
        if overrides:
            self.settings.update(
                {key: value for key, value in overrides.items() if value is not None}
            )
        self._session: Any = None
        self._input_name: str | None = None
        self._output_names: list[str] = []
        self._model_path: Path | None = None
        self._using_gpu = False
        self.runtime_warning = ""

    # Subclasses provide model-specific behavior.
    def prepare(self) -> None:
        raise NotImplementedError

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError

    def _postprocess_one(
        self, scores: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        raise NotImplementedError

    def _get_batch_size_cfg(self) -> int:
        value = self.settings.get("batch_size", 1)
        try:
            return max(1, min(64, int(value)))
        except (TypeError, ValueError):
            return 1

    def _device(self) -> str:
        value = str(self.settings.get("device") or "auto").strip().lower()
        return value if value in {"auto", "cpu", "cuda"} else "auto"

    def _selected_gpu_index(self) -> int | None:
        if self._device() != "cuda":
            return None
        value = self.settings.get("gpu_index")
        if value is None or value == "" or isinstance(value, bool):
            return None
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        return index if index >= 0 else None

    def _strict_cuda_device(self) -> bool:
        return self._selected_gpu_index() is not None

    def _create_session(self, model_path: Path) -> None:
        """创建 ORT session，并识别 GPU EP 静默降级。"""

        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LocalTaggingError(
                "未安装 onnxruntime；请安装 onnxruntime（CPU）或 onnxruntime-gpu（CUDA）"
            ) from exc

        available = set()
        try:
            available = set(ort.get_available_providers())
        except (AttributeError, RuntimeError):
            available = set()
        requested = self._device()
        gpu_ep: str | None = None
        if requested in {"auto", "cuda"} and "CUDAExecutionProvider" in available:
            gpu_ep = "CUDAExecutionProvider"
        elif requested == "auto" and "DmlExecutionProvider" in available:
            gpu_ep = "DmlExecutionProvider"

        if requested == "cuda" and gpu_ep is None:
            if self._strict_cuda_device():
                raise LocalTaggingError(
                    f"所选 GPU {self._selected_gpu_index()} 没有可用的 CUDAExecutionProvider"
                )
            self.runtime_warning = "未发现 CUDAExecutionProvider，已回退 CPU"
        provider_names = [gpu_ep, "CPUExecutionProvider"] if gpu_ep else ["CPUExecutionProvider"]
        providers: list[Any] = list(provider_names)
        if gpu_ep == "CUDAExecutionProvider":
            providers[0] = ("CUDAExecutionProvider", {"device_id": 0})
        try:
            context = (
                _silence_cuda_stderr()
                if gpu_ep == "CUDAExecutionProvider"
                else contextlib.nullcontext()
            )
            with context:
                session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as exc:  # noqa: BLE001 - provider libraries vary by platform
            if gpu_ep is None:
                raise LocalTaggingError(
                    f"无法创建 ONNX Runtime session：{_safe_error(exc)}"
                ) from exc
            if self._strict_cuda_device():
                raise LocalTaggingError(
                    f"所选 GPU {self._selected_gpu_index()} 初始化失败：{_safe_error(exc)}"
                ) from exc
            self.runtime_warning = (
                f"{gpu_ep} 初始化失败，已回退 CPU：{_safe_error(exc)}"
            )
            logger.warning("%s: %s", self.name, self.runtime_warning)
            try:
                session = ort.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
            except Exception as cpu_exc:  # noqa: BLE001
                raise LocalTaggingError(
                    f"GPU 和 CPU ONNX session 均创建失败：{_safe_error(cpu_exc)}"
                ) from cpu_exc

        actual = []
        try:
            actual = list(session.get_providers())
        except (AttributeError, RuntimeError):
            actual = list(provider_names)
        self._using_gpu = bool(gpu_ep and gpu_ep in actual)
        if gpu_ep and not self._using_gpu and self._strict_cuda_device():
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - fail with the device error
                    pass
            raise LocalTaggingError(
                f"所选 GPU {self._selected_gpu_index()} 未能启用 CUDAExecutionProvider"
            )
        if gpu_ep and not self._using_gpu and not self.runtime_warning:
            self.runtime_warning = f"{gpu_ep} 静默降级到 CPU"
            logger.warning(
                "%s: %s (actual=%s)", self.name, self.runtime_warning, actual
            )
        self._session = session
        self._model_path = Path(model_path)
        try:
            inputs = list(session.get_inputs())
            if not inputs:
                raise LocalTaggingError("ONNX 模型没有输入节点")
            self._input_name = str(getattr(inputs[0], "name", "input"))
            self._output_names = [
                str(getattr(item, "name", "")) for item in session.get_outputs()
            ]
        except Exception:
            self._clear_session()
            raise

    def _fallback_to_cpu_session(self) -> bool:
        if self._model_path is None:
            return False
        if self._strict_cuda_device():
            return False
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]

            model_path = self._model_path
            session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            inputs = list(session.get_inputs())
            input_name = (
                str(getattr(inputs[0], "name", "input")) if inputs else self._input_name
            )
            output_names = [
                str(getattr(item, "name", "")) for item in session.get_outputs()
            ]
            previous = self._session
            self._session = session
            self._input_name = input_name
            self._output_names = output_names
            self._using_gpu = False
            self.runtime_warning = "CUDA 推理失败，已回退 CPU"
            self._close_session_object(previous)
            return True
        except Exception:  # noqa: BLE001
            if "session" in locals() and session is not self._session:
                self._close_session_object(session)
            logger.exception("%s: CPU fallback failed", self.name)
            return False

    def _clear_session(self) -> None:
        """Drop a partially prepared session so a later call can retry."""

        session = self._session
        self._session = None
        self._input_name = None
        self._output_names = []
        self._model_path = None
        self._using_gpu = False
        self._close_session_object(session)

    def _close_session_object(self, session: Any) -> None:
        if session is None:
            return
        close = getattr(session, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                logger.debug(
                    "%s: failed to close ONNX session", self.name, exc_info=True
                )

    def close(self) -> None:
        """Release the current ORT session before the worker process exits."""

        self._clear_session()

    @staticmethod
    def _is_cuda_error(exc: BaseException) -> bool:
        text = str(exc).upper()
        return any(
            token in text
            for token in ("CUDA", "CUBLAS", "CUDNN", "OUT OF MEMORY", "OOM")
        )

    def _effective_batch_size(self) -> int:
        if self._session is not None:
            try:
                providers = set(self._session.get_providers())
            except (AttributeError, RuntimeError, TypeError):
                providers = set()
            using_cuda = "CUDAExecutionProvider" in providers
            self._using_gpu = bool(
                providers.intersection(
                    {"CUDAExecutionProvider", "DmlExecutionProvider"}
                )
            )
        else:
            using_cuda = False
        # DirectML has model/provider-specific batching behavior.  Keep the
        # conservative single-image contract there; CUDA is the only backend
        # for which the configured batch size is currently validated.
        return self._get_batch_size_cfg() if using_cuda else 1

    def _preprocess_one(
        self, indexed: tuple[int, Path]
    ) -> tuple[int, np.ndarray | None, str | None]:
        index, path = indexed
        try:
            with Image.open(path) as image:
                return index, self._preprocess(image), None
        except Exception as exc:  # noqa: BLE001 - one bad image must not kill a batch
            return index, None, _safe_error(exc)

    # Compatibility name used by the original tagger implementation and its
    # callers; keeping it as a method also makes it convenient to monkeypatch
    # in focused tests.
    def _preprocess_one_safe(
        self, indexed: tuple[int, Path]
    ) -> tuple[int, np.ndarray | None, str | None]:
        return self._preprocess_one(indexed)

    def tag(
        self, image_paths: list[Path], on_progress: ProgressFn = lambda _d, _t: None
    ) -> Iterator[TagResult]:
        if self._session is None:
            self.prepare()
        if self._session is None or self._input_name is None:
            raise LocalTaggingError("ONNX provider 尚未准备完成")
        paths = [Path(path) for path in image_paths]
        total = len(paths)
        batch_size = self._effective_batch_size()
        pool = (
            ThreadPoolExecutor(
                max_workers=batch_size, thread_name_prefix=f"{self.name}-prep"
            )
            if batch_size > 1
            else None
        )
        try:
            for start in range(0, total, batch_size):
                chunk = paths[start : start + batch_size]
                prepared: list[tuple[int, np.ndarray]] = []
                errors: dict[int, str] = {}
                values = (
                    pool.map(self._preprocess_one_safe, list(enumerate(chunk)))
                    if pool is not None
                    else map(self._preprocess_one_safe, list(enumerate(chunk)))
                )
                for index, array, error in values:
                    if error or array is None:
                        errors[index] = error or "图片预处理失败"
                    else:
                        prepared.append((index, array))

                logits: np.ndarray | None = None
                if prepared:
                    batch: np.ndarray | None = None
                    try:
                        batch = np.stack(
                            [array for _index, array in prepared], axis=0
                        ).copy()
                        raw = self._session.run(
                            self._output_names or None, {self._input_name: batch}
                        )[0]
                        logits = _normalize_output(raw, len(prepared))
                    except Exception as exc:  # noqa: BLE001
                        if self._strict_cuda_device() and self._is_cuda_error(exc):
                            raise LocalTaggingError(
                                f"所选 GPU {self._selected_gpu_index()} 推理失败：{_safe_error(exc)}"
                            ) from exc
                        if (
                            batch is not None
                            and self._is_cuda_error(exc)
                            and self._fallback_to_cpu_session()
                        ):
                            try:
                                raw = self._session.run(
                                    self._output_names or None,
                                    {self._input_name: batch},
                                )[0]
                                logits = _normalize_output(raw, len(prepared))
                            except Exception as retry_exc:  # noqa: BLE001
                                errors.update(
                                    {
                                        index: f"模型推理失败：{_safe_error(retry_exc)}"
                                        for index, _ in prepared
                                    }
                                )
                        else:
                            errors.update(
                                {
                                    index: f"模型推理失败：{_safe_error(exc)}"
                                    for index, _ in prepared
                                }
                            )

                prepared_pos = 0
                for index, path in enumerate(chunk):
                    if index in errors:
                        result: TagResult = {
                            "image": path,
                            "tags": [],
                            "error": errors[index],
                        }
                    elif logits is not None and prepared_pos < len(logits):
                        try:
                            tags, raw_scores = self._postprocess_one(
                                logits[prepared_pos]
                            )
                            result = {
                                "image": path,
                                "tags": tags,
                                "raw_scores": raw_scores,
                            }
                        except Exception as exc:  # noqa: BLE001
                            result = {
                                "image": path,
                                "tags": [],
                                "error": f"标签后处理失败：{_safe_error(exc)}",
                            }
                        prepared_pos += 1
                    else:
                        result = {
                            "image": path,
                            "tags": [],
                            "error": "模型没有返回有效结果",
                        }
                    yield result
                    on_progress(start + index + 1, total)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)


def _normalize_output(value: Any, batch_size: int) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("模型输出 batch 大小无效")
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("模型输出为空")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("模型输出必须是数值数组")
    if array.ndim == 0:
        if batch_size != 1:
            raise ValueError(f"模型输出 batch 维度不匹配：scalar/{batch_size}")
        return array.reshape(1, 1)
    if array.ndim == 1:
        if batch_size == 1:
            return array.reshape(1, -1)
        # A one-dimensional multi-image output can only unambiguously mean
        # one scalar per image.  Do not reinterpret a flattened label matrix
        # as a batch merely because its length is divisible by batch_size.
        if array.shape[0] == batch_size:
            return array.reshape(batch_size, 1)
        raise ValueError(f"模型输出 batch 维度不匹配：{array.shape}")
    if array.shape[0] != batch_size:
        raise ValueError(f"模型输出 batch 维度不匹配：{array.shape}")
    return array.reshape(batch_size, -1)


def _safe_error(exc: BaseException) -> str:
    return " ".join(str(exc or "").split())[:500] or exc.__class__.__name__


__all__ = ["OnnxTaggerBase", "onnxruntime_status", "silenced_fd_stderr"]

# Public compatibility alias for integrations that used the reference
# implementation's stderr context manager.
silenced_fd_stderr = _silence_cuda_stderr
