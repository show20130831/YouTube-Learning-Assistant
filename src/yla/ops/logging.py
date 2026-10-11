"""Logging with the current run ID on every line, so one run's logs can be followed end to end."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_run: ContextVar[str] = ContextVar("current_run", default="-")
FORMAT = "%(asctime)s %(levelname)s run=%(run_id)s %(name)s: %(message)s"


class RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _current_run.get()
        return True


@contextmanager
def run_context(run_id: int | str) -> Iterator[None]:
    token = _current_run.set(str(run_id))
    try:
        yield
    finally:
        _current_run.reset(token)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(FORMAT))
    handler.addFilter(RunIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
