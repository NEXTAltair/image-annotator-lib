"""Local backend ownership and context leases shared across frameworks."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

from PIL import Image

from ...exceptions.errors import OutOfMemoryError
from ..model_factory import ModelLoad
from ..utils import logger
from .annotator import BaseAnnotator


def local_model_enter[AnnotatorT: LocalModelAnnotator](
    method: Callable[[AnnotatorT], AnnotatorT],
) -> Callable[[AnnotatorT], AnnotatorT]:
    """Reserve the owner before preparation, counting a superclass call only once.

    The dispatch marker lasts only for this method call, so separate nested ``with``
    blocks still acquire separate leases. Backend subclasses overriding a context
    method should apply the matching decorator to include their preparation.
    """

    @wraps(method)
    def enter(self: AnnotatorT) -> AnnotatorT:
        with ModelLoad._RESOURCE_LOCK:
            if getattr(self, "_context_entry_dispatch", False):
                return method(self)

            # A rejected replacement must not run the incoming owner's cleanup.
            ModelLoad.begin_model_context(self.model_name, self)
            self._context_entry_dispatch = True
            try:
                if ModelLoad.is_final_model_context(self.model_name, self):
                    ModelLoad.register_component_releaser(
                        self.model_name, self._release_retained_components, owner=self
                    )
                result = method(self)
                ModelLoad.finish_model_entry(self.model_name, self)
                return result
            except BaseException:
                ModelLoad.end_model_context(self.model_name, self, failed=True)
                raise
            finally:
                self._context_entry_dispatch = False

    return enter


def local_model_exit[AnnotatorT: LocalModelAnnotator](
    method: Callable[[AnnotatorT, type[BaseException] | None, BaseException | None, Any], None],
) -> Callable[[AnnotatorT, type[BaseException] | None, BaseException | None, Any], None]:
    """Run backend cleanup only on the owner's final exit, then return its lease."""

    @wraps(method)
    def exit_context(
        self: AnnotatorT,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        with ModelLoad._RESOURCE_LOCK:
            if getattr(self, "_context_exit_dispatch", False):
                method(self, exc_type, exc_val, exc_tb)
                return

            # Direct __exit__ calls have no lease to return or owner to validate.
            if not ModelLoad.is_model_active(self.model_name):
                method(self, exc_type, exc_val, exc_tb)
                return

            final_exit = ModelLoad.begin_model_exit(self.model_name, self)
            self._context_exit_dispatch = True
            failed = exc_type is not None
            try:
                if final_exit:
                    method(self, exc_type, exc_val, exc_tb)
            except BaseException:
                failed = True
                raise
            finally:
                self._context_exit_dispatch = False
                ModelLoad.end_model_context(self.model_name, self, failed=failed)

    return exit_context


class LocalModelAnnotator(BaseAnnotator):
    """Shared retained-reference cleanup and OOM handling for local backends."""

    _context_entry_dispatch = False
    _context_exit_dispatch = False
    _prepared = False

    def _release_retained_components(self) -> None:
        """Drop actual references when the loader releases this owner."""
        self._prepared = False
        components = self.components
        self.components = None
        if components:
            cast(dict[str, Any], components).clear()

    def _discard_components(self) -> None:
        """Invalidate only this owner's unusable generation, keeping its leases."""
        with ModelLoad._RESOURCE_LOCK:
            try:
                ModelLoad.invalidate_model(self.model_name, self)
            except Exception as error:
                logger.error(f"コンポーネント解放失敗 ({self.model_name}): {error}")
            finally:
                self._release_retained_components()

    def _execute_pipeline(self, images: list[Image.Image]) -> list:
        """Invalidate OOM resources before predict converts the failure to a result."""
        try:
            return super()._execute_pipeline(images)
        except Exception as error:
            if not _is_memory_failure(error):
                raise
            self._discard_components()
            raise OutOfMemoryError(
                f"モデル '{self.model_name}' の処理中にメモリ不足が発生しました。"
            ) from error


def _is_memory_failure(error: BaseException) -> bool:
    """Recognize native OOM causes without importing other backend runtimes."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OutOfMemoryError | MemoryError):
            return True
        for error_class in type(current).__mro__:
            if error_class.__module__.startswith("torch") and error_class.__name__ == "OutOfMemoryError":
                return True
            if (
                error_class.__module__.startswith("tensorflow")
                and error_class.__name__ == "ResourceExhaustedError"
            ):
                return True
        current = current.__cause__ or current.__context__
    return False
