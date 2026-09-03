"""协议和错误类型 for local tagging providers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Protocol, TypedDict, runtime_checkable


ProgressFn = Callable[[int, int], None]


class TagResult(TypedDict, total=False):
    """单张图片的本地推理结果。"""

    image: Path
    tags: list[str]
    caption: str
    raw_scores: dict[str, float]
    error: str


class LocalTaggingError(RuntimeError):
    """本地 provider 无法初始化或推理时的用户可见错误。"""


@runtime_checkable
class Tagger(Protocol):
    name: str
    requires_service: bool

    def is_available(self) -> tuple[bool, str]: ...

    def prepare(self) -> None: ...

    def close(self) -> None: ...

    def tag(
        self,
        image_paths: list[Path],
        on_progress: ProgressFn = lambda _done, _total: None,
    ) -> Iterator[TagResult]: ...


__all__ = ["LocalTaggingError", "ProgressFn", "TagResult", "Tagger"]
