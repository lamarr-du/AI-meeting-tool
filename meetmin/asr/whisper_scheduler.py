"""进程内复用 Whisper 权重，在原生编码窗之间调度桌面转写。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import IntEnum
from threading import Condition, local
from time import monotonic
from typing import Any
from weakref import WeakMethod


class Priority(IntEnum):
    BACKGROUND = 0
    LIVE = 1
    CATCH_UP = 2


class SchedulingError(RuntimeError):
    """The recording can continue, but its current live decode cannot."""


def background_priority() -> Priority:
    return Priority.BACKGROUND


class WaitTime:
    """记录协调器真正等待模型或推理窗口的时间，供停录尾窗计算时限。"""

    def __init__(self) -> None:
        self._state: tuple[float, float | None] = (0.0, None)

    @contextmanager
    def waiting(self) -> Iterator[None]:
        total, _ = self._state
        started = monotonic()
        self._state = total, started
        try:
            yield
        finally:
            self._state = total + monotonic() - started, None

    def seconds(self) -> float:
        total, started = self._state
        return total if started is None else total + monotonic() - started


@dataclass
class ModelLease:
    key: tuple[str, str, str]
    model: Any
    users: int = 1
    background_users: int = 0
    valid: bool = True


class WhisperScheduler:
    def __init__(self) -> None:
        self._condition = Condition()
        self._slot: ModelLease | None = None
        self._loading = False
        self._pending: list[Callable[[], Priority]] = []
        self._busy = False
        self._last_was_live = False
        self._local = local()

    def _wait(self, clock: WaitTime | None) -> None:
        with clock.waiting() if clock is not None else nullcontext():
            self._condition.wait(0.05)

    def acquire_model(
        self,
        key: tuple[str, str, str],
        factory: Callable[[], Any],
        check: Callable[[], None],
        clock: WaitTime | None,
        *,
        live: bool,
    ) -> ModelLease:
        with self._condition:
            while True:
                check()
                slot = self._slot
                if slot is None and not self._loading:
                    self._loading = True
                    break
                if slot is not None and slot.key == key and slot.valid:
                    slot.users += 1
                    slot.background_users += not live
                    return slot
                if slot is not None and not live and not slot.background_users and slot.valid:
                    # An older offline job can start after a newer live job opened the model.
                    slot.valid = False
                    self._condition.notify_all()
                self._wait(clock)

        try:
            model = factory()
            original = WeakMethod(model.encode)

            def encode(features: Any) -> Any:
                # faster-whisper can advance several windows without yielding a segment.
                # Only the encoder sees every window, including skipped silence.
                self._local.checkpoint()
                method = original()
                assert method is not None
                return method(features)

            model.encode = encode
            slot = ModelLease(key, model, background_users=int(not live))
            with self._condition:
                self._slot = slot
                return slot
        finally:
            with self._condition:
                self._loading = False
                self._condition.notify_all()

    def release_model(self, slot: ModelLease, *, live: bool) -> None:
        with self._condition:
            slot.users -= 1
            slot.background_users -= not live
            if not slot.users:
                slot.model = None
                self._slot = None
            self._condition.notify_all()

    def invalidate_model(self, slot: ModelLease, *, preserve_shared: bool) -> bool:
        with self._condition:
            if preserve_shared and slot.users > 1:
                return False
            slot.valid = False
            self._condition.notify_all()
            return True

    def _next(self) -> Callable[[], Priority]:
        background = live = None
        for request in self._pending:
            priority = request()
            if priority == Priority.CATCH_UP:
                return request
            if priority == Priority.LIVE and live is None:
                live = request
            elif priority == Priority.BACKGROUND and background is None:
                background = request
        if live is not None and (background is None or not self._last_was_live):
            return live
        assert background is not None
        return background

    @contextmanager
    def windows(
        self,
        priority: Callable[[], Priority],
        check: Callable[[], None],
        clock: WaitTime | None,
    ) -> Iterator[None]:
        held = False

        def checkpoint() -> None:
            nonlocal held
            with self._condition:
                if held:
                    self._busy = held = False

                # Each context has its own request identity, even for the same role.
                def request() -> Priority:
                    return priority()

                self._pending.append(request)
                self._condition.notify_all()
                try:
                    while True:
                        check()
                        if not self._busy and self._next() is request:
                            self._last_was_live = request() != Priority.BACKGROUND
                            self._busy = held = True
                            break
                        self._wait(clock)
                finally:
                    self._pending.remove(request)
                    self._condition.notify_all()

        self._local.checkpoint = checkpoint
        try:
            check()
            yield
        finally:
            del self._local.checkpoint
            if held:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()
