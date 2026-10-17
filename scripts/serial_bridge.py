# -*- coding: utf-8 -*-
"""Serial data bridge: send TEPAD simulation data to the ``sim_ui/`` MFC host application
as ``tick,G,P`` CSV frames.

Frame protocol
--------------
One frame per line, UTF-8 / ASCII, terminated by ``\\n``::

    tick,G,P

* ``tick``: increasing integer index (starting at 0);
* ``G``: normalized irradiance = ``clip(irradiance / 1000, 0, 1)``, exactly the ``g``
  passed to the detector by ``src/simulation/model.py::run_detector``;
* ``P``: normalized AC power (0~1), i.e. ``SimulatedDay.power``.

Commands sent in the opposite direction (manual commands from the host application, such
as ``PING`` / ``SPEED 0.25``) are read and printed by this script: ``SPEED <seconds>``
changes the frame interval (replay speed) immediately, anything else is only logged.

Usage
-----
    # real serial port (requires pyserial: pip install pyserial)
    python scripts/serial_bridge.py --port COM4 --baud 115200

    # without hardware: TCP server (select TCP as host port type, connect to 127.0.0.1:18801)
    python scripts/serial_bridge.py --tcp --host 127.0.0.1 --tcp-port 18801

    # text frames only, to inspect the data itself
    python scripts/serial_bridge.py --out frames.csv --ticks 720

    # inject a 3-sigma spike at tick 300 and 50% shading at tick 400
    python scripts/serial_bridge.py --tcp --inject-spike-tick 300 --inject-spike-sigma 3 \
        --inject-shade-start 400 --inject-shade-duration 60 --inject-shade-factor 0.5
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.simulation.model import (  # noqa: E402
    TrueParameters,
    inject_shading,
    inject_spike,
    simulate_normal_day,
)

DEFAULT_TCP_PORT = 18801
#: Decimals per frame (6 are enough for display and detection, and do not slow down low-baud ports)
FRAME_PRECISION = 6
#: Seconds between the half-close after the last frame and the real close, giving the host
#: application time to drain the tail frames
CLOSE_GRACE_SEC = 1.0
#: Allowed frame-interval range (seconds) for the SPEED command: the lower bound keeps the
#: peer from being flooded, the upper bound keeps the UI from looking frozen
MIN_INTERVAL_SEC = 0.002
MAX_INTERVAL_SEC = 5.0
#: Length of a single sleep slice, so a SPEED change takes effect without waiting a full frame
PACE_SLICE_SEC = 0.05


class Pace:
    """Frame-interval holder: the sender loop reads it per frame, the host's ``SPEED`` command writes it.

    It is a separate class so that the sending thread (reading the pace) and the receiving
    thread (changing the pace) do not block each other, without passing interval down as a
    parameter everywhere.
    """

    def __init__(self, interval: float) -> None:
        self._interval = self._clamp(interval)
        self._lock = threading.Lock()

    @staticmethod
    def _clamp(seconds: float) -> float:
        return min(max(float(seconds), MIN_INTERVAL_SEC), MAX_INTERVAL_SEC)

    def get(self) -> float:
        with self._lock:
            return self._interval

    def set(self, seconds: float) -> float:
        with self._lock:
            self._interval = self._clamp(seconds)
            return self._interval


def build_frames(seed: int, ticks: int, spike_tick: int, spike_sigma: float,
                 shade_start: int, shade_duration: int,
                 shade_factor: float) -> List[Tuple[int, float, float]]:
    """Build the ``tick,G,P`` frame sequence of one simulated day, optionally injecting a
    spike / shading."""
    theta = TrueParameters()
    day = simulate_normal_day(seed, ticks, theta)
    g = np.clip(day.irradiance / 1000.0, 0.0, 1.0)
    p = day.power.copy()
    if spike_tick >= 0:
        p = inject_spike(p, spike_tick, spike_sigma * float(day.sigma[spike_tick]))
        print("[inject] spike tick={} amplitude={:.2f}sigma".format(spike_tick, spike_sigma))
    if shade_start >= 0:
        p = inject_shading(p, shade_start, shade_duration, shade_factor)
        print("[inject] shading start={} duration={} factor={:.2f}".format(
            shade_start, shade_duration, shade_factor))
    return [(int(i), float(g[i]), float(p[i])) for i in range(len(g))]


def iter_stream(frames: List[Tuple[int, float, float]], pace: Pace,
                loop: bool) -> Iterator[str]:
    """Render the frames into CSV text lines at the configured pace; the pace can be changed
    at runtime by the ``SPEED`` command."""
    while True:
        for tick, g, p in frames:
            yield "{:d},{:.{prec}f},{:.{prec}f}\n".format(tick, g, p, prec=FRAME_PRECISION)
            interval = pace.get()
            remaining = interval
            while remaining > 0:
                step = min(remaining, PACE_SLICE_SEC)
                time.sleep(step)
                latest = pace.get()
                if latest != interval:
                    # The speed was just changed: restart the timing from zero with the new
                    # interval (compare against the current pace, not the remaining time,
                    # otherwise intervals spanning several slices would never finish sleeping)
                    interval = latest
                    remaining = latest
                else:
                    remaining -= step
        if not loop:
            return


def reader_thread(name: str, stream, pace: Pace) -> None:
    """Read lines sent by the host application in the background: ``SPEED <seconds>`` changes
    the replay pace, anything else is only printed."""
    while True:
        try:
            line = stream.readline()
        except OSError:
            break
        if not line:
            break
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("SPEED"):
            try:
                seconds = float(line.split(None, 1)[1])
            except (IndexError, ValueError):
                print("[recv {}] {} -> cannot parse SPEED, correct form: SPEED <seconds>".format(name, line))
                continue
            applied = pace.set(seconds)
            print("[speed] replay interval set to {:.3f}s/tick (about {:.0f} tick/s)".format(
                applied, 1.0 / applied))
            continue
        print("[recv {}] {}".format(name, line))


def send_serial(args: argparse.Namespace, frames, pace: Pace) -> None:
    try:
        import serial  # type: ignore
    except ImportError:
        sys.exit("pyserial is not installed, --port is unavailable. Run first: pip install pyserial")
    with serial.Serial(args.port, args.baud, timeout=0.2) as port:
        print("[serial] {} @ {} 8N1 opened".format(args.port, args.baud))
        threading.Thread(target=reader_thread, args=(args.port, port, pace),
                         daemon=True).start()
        for line in frames:
            port.write(line.encode("ascii"))
            port.flush()
        print("[serial] frame sequence sent")


def send_tcp(args: argparse.Namespace, frames, pace: Pace) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.tcp_port))
    server.listen(1)
    print("[tcp] listening on {}:{}, waiting for the host application...".format(args.host, args.tcp_port))
    conn, addr = server.accept()
    print("[tcp] host application connected: {}".format(addr))
    with conn:
        threading.Thread(target=reader_thread, args=(addr, conn.makefile("r"), pace),
                         daemon=True).start()
        for line in frames:
            try:
                conn.sendall(line.encode("ascii"))
            except OSError:
                print("[tcp] connection lost, stopping transmission")
                return
        # Half-close before leaving the with block: on Windows, close() sends RST when data
        # is still pending in the send buffer, so the host application loses the tail frames
        # and reports 10054; shutdown(SHUT_WR) guarantees an orderly FIN after all frames.
        try:
            conn.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        time.sleep(CLOSE_GRACE_SEC)
    print("[tcp] frame sequence sent")


def send_file(args: argparse.Namespace, frames) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii", newline="\n") as fh:
        fh.write("tick,G,P\n")
        for line in frames:
            fh.write(line)
    print("[file] frame sequence written: {}".format(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="TEPAD simulation data serial/network bridge")
    ap.add_argument("--port", default=None, help="serial port, e.g. COM4 (requires pyserial)")
    ap.add_argument("--baud", type=int, default=115200, help="baud rate, default 115200")
    ap.add_argument("--tcp", action="store_true", help="use a TCP server instead (for integration without hardware)")
    ap.add_argument("--host", default="127.0.0.1", help="TCP listen address")
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT, help="TCP listen port")
    ap.add_argument("--out", default=None, help="write the frames to a CSV file (do not send)")
    ap.add_argument("--seed", type=int, default=config.SIMULATION_SEED + 700000,
                    help="simulation-day seed (offset from the calibration-day seed)")
    ap.add_argument("--ticks", type=int, default=config.SIMULATION_DAY_TICKS,
                    help="number of ticks to send")
    ap.add_argument("--interval", type=float, default=0.05,
                    help="initial frame interval in seconds (can be changed at runtime by the host's SPEED command)")
    ap.add_argument("--loop", action="store_true", help="resend in a loop (to demonstrate long-running operation)")
    ap.add_argument("--inject-spike-tick", type=int, default=-1, help="spike injection tick, <0 disables")
    ap.add_argument("--inject-spike-sigma", type=float, default=3.0, help="spike amplitude (multiples of sigma)")
    ap.add_argument("--inject-shade-start", type=int, default=-1, help="shading start tick, <0 disables")
    ap.add_argument("--inject-shade-duration", type=int, default=60, help="shading duration in ticks")
    ap.add_argument("--inject-shade-factor", type=float, default=0.5, help="shading attenuation factor (0,1)")
    args = ap.parse_args()

    frames = build_frames(args.seed, args.ticks, args.inject_spike_tick,
                          args.inject_spike_sigma, args.inject_shade_start,
                          args.inject_shade_duration, args.inject_shade_factor)
    print("[data] {} frames, G∈[{:.3f},{:.3f}], P∈[{:.3f},{:.3f}]".format(
        len(frames),
        min(f[1] for f in frames), max(f[1] for f in frames),
        min(f[2] for f in frames), max(f[2] for f in frames)))

    pace = Pace(args.interval)
    stream = iter_stream(frames, pace, args.loop)
    if args.out:
        send_file(args, stream)
    elif args.tcp:
        send_tcp(args, stream, pace)
    elif args.port:
        send_serial(args, stream, pace)
    else:
        # Default output to stdout, for piping / visual inspection
        for line in stream:
            sys.stdout.write(line)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
