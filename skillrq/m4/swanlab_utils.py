"""Small SwanLab logging wrapper for M4 training and inference."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class SwanLabLogger:
    def __init__(
        self,
        project: str | None,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        self.enabled = bool(project)
        self._module = None
        if not self.enabled:
            return
        try:
            import swanlab  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SwanLab logging is enabled but `swanlab` is not installed. "
                "Install M4 requirements with `uv pip install -r skillrq/m4/requirements.txt`, "
                "or pass `--disable-swanlab`."
            ) from exc
        self._module = swanlab
        kwargs: dict[str, Any] = {"project": project, "config": dict(config or {})}
        if run_name:
            kwargs["experiment_name"] = run_name
        if tags:
            kwargs["tags"] = list(tags)
        _init_swanlab(swanlab, kwargs)

    def log(self, values: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._module is None:
            return
        payload = {key: value for key, value in values.items() if _is_loggable(value)}
        if not payload:
            return
        try:
            self._module.log(payload, step=step)
        except TypeError:
            self._module.log(payload)

    def finish(self) -> None:
        if not self.enabled or self._module is None:
            return
        finish = getattr(self._module, "finish", None)
        if callable(finish):
            finish()


def _is_loggable(value: Any) -> bool:
    return value is None or isinstance(value, (int, float, str, bool))


def _init_swanlab(swanlab: Any, kwargs: dict[str, Any]) -> None:
    attempts = [dict(kwargs)]
    if "experiment_name" in kwargs:
        renamed = dict(kwargs)
        renamed["name"] = renamed.pop("experiment_name")
        attempts.append(renamed)
    for key_to_drop in ("tags", "name", "experiment_name"):
        if key_to_drop in kwargs:
            stripped = {key: value for key, value in kwargs.items() if key != key_to_drop}
            attempts.append(stripped)
    last_error: TypeError | None = None
    for attempt in attempts:
        try:
            swanlab.init(**attempt)
            return
        except TypeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
