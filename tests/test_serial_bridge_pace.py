"""Regression tests for the replay pacing of ``scripts/serial_bridge.py``.

The tests drive ``iter_stream`` with a fake clock, so the whole suite finishes in
milliseconds and never actually sleeps. The fake clock has a cap: if the pacing logic
ever gets stuck in an infinite loop (it once compared "remaining time" instead of the
"current pace", so intervals with ``interval > PACE_SLICE_SEC`` never finished sleeping
and no further frame was emitted), ``_VirtualTimeLimit`` is raised to fail the test
instead of hanging.
"""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "serial_bridge", ROOT / "scripts" / "serial_bridge.py")
serial_bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serial_bridge)


class _VirtualTimeLimit(Exception):
    """Virtual time exceeded the budget: the send loop did not wake up on schedule."""


class _FakeClock:
    def __init__(self, cap_seconds: float) -> None:
        self.now = 0.0
        self.cap_seconds = cap_seconds

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.now > self.cap_seconds:
            raise _VirtualTimeLimit(
                "virtual time passed {:.1f}s; the pacing loop never yielded a frame".format(
                    self.cap_seconds))


def _frames(count=100):
    return [(i, 0.5, 0.4) for i in range(count)]


class PaceTests(unittest.TestCase):
    def test_interval_is_clamped_to_a_sane_range(self):
        self.assertEqual(serial_bridge.Pace(0.0).get(),
                         serial_bridge.MIN_INTERVAL_SEC)
        self.assertEqual(serial_bridge.Pace(9999.0).get(),
                         serial_bridge.MAX_INTERVAL_SEC)
        self.assertAlmostEqual(serial_bridge.Pace(0.1).get(), 0.1)

    def test_set_returns_the_clamped_value(self):
        pace = serial_bridge.Pace(0.1)
        self.assertEqual(pace.set(-1.0), serial_bridge.MIN_INTERVAL_SEC)
        self.assertEqual(pace.set(0.25), 0.25)


class IterStreamTests(unittest.TestCase):
    def _consume(self, gen, count):
        for _ in range(count):
            next(gen)

    def test_frame_text_has_no_stray_space(self):
        pace = serial_bridge.Pace(0.1)
        line = next(serial_bridge.iter_stream(_frames(1), pace, loop=False))
        self.assertEqual(line, "0,0.500000,0.400000\n")

    def test_constant_pacing_yields_one_frame_per_interval(self):
        # An interval of 0.1s is larger than the 0.05s slice: this is the path that used to hang.
        pace = serial_bridge.Pace(0.1)
        clock = _FakeClock(cap_seconds=20.0)
        with mock.patch.object(serial_bridge.time, "sleep", clock.sleep):
            self._consume(serial_bridge.iter_stream(_frames(100), pace, loop=False), 40)
        self.assertAlmostEqual(clock.now, 3.9, places=3)

    def test_small_interval_still_paces_correctly(self):
        pace = serial_bridge.Pace(0.02)
        clock = _FakeClock(cap_seconds=20.0)
        with mock.patch.object(serial_bridge.time, "sleep", clock.sleep):
            self._consume(serial_bridge.iter_stream(_frames(100), pace, loop=False), 10)
        self.assertAlmostEqual(clock.now, 0.18, places=3)

    def test_speed_change_takes_effect_at_the_next_frame(self):
        pace = serial_bridge.Pace(0.1)
        clock = _FakeClock(cap_seconds=20.0)
        gen = serial_bridge.iter_stream(_frames(100), pace, loop=False)
        with mock.patch.object(serial_bridge.time, "sleep", clock.sleep):
            self._consume(gen, 10)          # 9 intervals of 0.1s
            pace.set(0.5)                   # the next interval immediately uses 0.5s
            self._consume(gen, 5)           # 5 intervals of 0.5s
        self.assertAlmostEqual(clock.now, 0.9 + 5 * 0.5, places=3)

    def test_stream_stops_when_loop_is_off(self):
        pace = serial_bridge.Pace(0.1)
        clock = _FakeClock(cap_seconds=20.0)
        with mock.patch.object(serial_bridge.time, "sleep", clock.sleep):
            lines = list(serial_bridge.iter_stream(_frames(3), pace, loop=False))
        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()
