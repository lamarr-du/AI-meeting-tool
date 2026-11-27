from concurrent.futures import ThreadPoolExecutor
from threading import Event
from time import monotonic, sleep

import pytest

from meetmin.asr.whisper_scheduler import (
    Priority,
    SchedulingError,
    WaitTime,
    WhisperScheduler,
    background_priority,
)


def check():
    pass


def wait_until_waiting(clock):
    deadline = monotonic() + 3
    while clock.seconds() == 0 and monotonic() < deadline:
        sleep(0.002)
    assert clock.seconds() > 0


@pytest.mark.parametrize(
    "priority, expected",
    [
        (Priority.LIVE, ["A1", "B1", "A2", "B2"]),
        (Priority.CATCH_UP, ["A1", "B1", "B2", "A2"]),
    ],
)
def test_shared_model_switches_at_every_encoder_window(priority, expected):
    scheduler = WhisperScheduler()
    first_window, release = Event(), Event()
    order = []

    class Model:
        def encode(self, label):
            order.append(label)
            if label == "A1":
                first_window.set()
                assert release.wait(5)

    key = ("same-model", "cuda", "int8_float32")
    a = scheduler.acquire_model(key, Model, check, None, live=False)

    def never_reload():
        pytest.fail("both sessions should reuse the same weights")

    b = scheduler.acquire_model(key, never_reload, check, None, live=True)
    assert a is b

    def offline():
        with scheduler.windows(background_priority, check, None):
            a.model.encode("A1")
            a.model.encode("A2")

    def live(label, clock):
        with scheduler.windows(lambda: priority, check, clock):
            b.model.encode(label)

    with ThreadPoolExecutor(max_workers=3) as workers:
        try:
            jobs = [workers.submit(offline)]
            assert first_window.wait(3)
            for label in ("B1", "B2"):
                clock = WaitTime()
                jobs.append(workers.submit(live, label, clock))
                wait_until_waiting(clock)
        finally:
            release.set()
        for job in jobs:
            job.result(timeout=5)
    assert order == expected
    scheduler.release_model(a, live=False)
    scheduler.release_model(b, live=True)
    assert b.model is None


def test_incompatible_live_model_wait_can_be_cancelled():
    scheduler = WhisperScheduler()
    stopped = Event()
    clock = WaitTime()

    class Model:
        def encode(self, _features):
            pass

    offline_key = ("offline", "cuda", "int8_float32")
    live_key = ("live", "cuda", "int8_float32")
    a = scheduler.acquire_model(offline_key, Model, check, None, live=False)

    def check_cancelled():
        if stopped.is_set():
            raise SchedulingError("overflow")

    def never_load():
        pytest.fail("an incompatible model must wait for the current lease")

    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(
            scheduler.acquire_model, live_key, never_load, check_cancelled, clock, live=True
        )
        try:
            wait_until_waiting(clock)
        finally:
            stopped.set()
        with pytest.raises(SchedulingError, match="overflow"):
            future.result(timeout=3)
    assert a.valid
    scheduler.release_model(a, live=False)
    b = scheduler.acquire_model(live_key, Model, check, None, live=True)
    scheduler.release_model(b, live=True)


def test_failed_live_share_keeps_the_offline_model_available():
    class Model:
        def encode(self, _features):
            pass

    scheduler = WhisperScheduler()
    key = ("model", "cuda", "int8_float32")
    a = scheduler.acquire_model(key, Model, check, None, live=False)
    b = scheduler.acquire_model(key, Model, check, None, live=True)
    assert not scheduler.invalidate_model(b, preserve_shared=True)
    assert a.valid
    assert scheduler.invalidate_model(a, preserve_shared=False)
    assert not b.valid
    scheduler.release_model(b, live=True)
    scheduler.release_model(a, live=False)
