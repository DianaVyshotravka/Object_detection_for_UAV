"""ClockSync -- MAVLink TIMESYNC offset between our clock and the autopilot's.

Runs as a background timer thread, not a tight loop.

IMPORTANT -- this class never receives. ``mavutil`` connections are not thread
safe, and two threads calling ``recv_match()`` on one connection steal each
other's messages: TIMESYNC replies disappear into the telemetry reader and
GLOBAL_POSITION_INT disappears here. So exactly one thread (MavlinkClient.run)
ever reads, and it hands TIMESYNC replies to ``handle_timesync()``.

This module imports no pymavlink, which is also what makes it unit-testable on
a machine with no autopilot and no pymavlink installed.
"""

import threading
import time
from typing import Callable, Optional


class ClockSync:
    def __init__(
        self,
        send_timesync: Callable[[int, int], None],
        stop_event: threading.Event,
        logger,
        interval_s: float = 10.0,
        max_rtt_s: float = 0.5,
    ):
        self.send_timesync = send_timesync  # (tc1, ts1) -> None
        self.stop_event = stop_event
        self.logger = logger
        self.interval_s = interval_s
        self.max_rtt_s = max_rtt_s

        self._lock = threading.Lock()
        self._offset_s: Optional[float] = None
        self._utc_offset_s: Optional[float] = None
        self._ts1 = 0

    # --- offset convention -------------------------------------------------
    # offset = our_clock - autopilot_clock, so:
    #     local_wall_time = autopilot_time + offset
    # Telemetry carries the autopilot's time-since-boot, and this is what turns
    # it into our wall clock so frames and telemetry can be compared. Getting
    # the sign backwards doubles the error instead of cancelling it.

    def get_offset(self) -> Optional[float]:
        """Non-blocking read. None until the first good round trip."""
        with self._lock:
            return self._offset_s

    # --- wall-clock UTC, a separate problem ---------------------------------
    # TIMESYNC tc1 carries the autopilot's time since BOOT, not UTC, so the
    # offset above cannot date a photo. The Pi has no battery RTC, so its own
    # clock is whatever it was at shutdown until NTP lands -- which never
    # happens on a drone with no network. SYSTEM_TIME.time_unix_usec is GPS
    # derived and is the only true UTC onboard.

    def set_utc_offset(self, unix_s: float, recv_wall_s: float) -> None:
        """From SYSTEM_TIME: how far our wall clock is from GPS UTC."""
        if unix_s <= 0:
            return  # autopilot has no GPS fix yet
        with self._lock:
            self._utc_offset_s = unix_s - recv_wall_s

    def to_utc(self, local_s: float) -> float:
        """Local wall-clock seconds -> UTC. Falls back to the Pi clock."""
        with self._lock:
            return local_s + (self._utc_offset_s or 0.0)

    def bootstrap(self, t_boot_s: float, recv_wall_s: float) -> None:
        """Crude cold-start offset from the first telemetry message.

        Without this, every frame is ungeotagged until the first TIMESYNC round
        trip completes -- up to a full interval of lost flight. Ignores link
        latency (tens of ms); TIMESYNC replaces it as soon as it lands.
        """
        with self._lock:
            if self._offset_s is None:
                self._offset_s = recv_wall_s - t_boot_s
                self.logger.log_info("clock: bootstrapped offset %.3f s from telemetry"
                                     % self._offset_s)

    def handle_timesync(self, msg, recv_ns: int) -> None:
        """Called by the MAVLink reader thread for every TIMESYNC message."""
        if msg.tc1 == 0:
            return  # tc1 == 0 means REQUEST, not response (incl. our own echo off a router)
        if msg.ts1 != self._ts1:
            return  # reply to someone else's exchange
        rtt_ns = recv_ns - msg.ts1
        if rtt_ns < 0 or rtt_ns > self.max_rtt_s * 1e9:
            self.logger.log_warning("TIMESYNC: rtt %.1f ms rejected" % (rtt_ns / 1e6))
            return  # one bad sample must not poison the offset
        offset_s = ((msg.ts1 + recv_ns) // 2 - msg.tc1) / 1e9
        with self._lock:
            self._offset_s = offset_s

    def _sync_once(self) -> None:
        self._ts1 = time.time_ns()
        try:
            self.send_timesync(0, self._ts1)  # tc1=0 -> this is a request
        except Exception as exc:  # a dropped link must not kill the thread
            self.logger.log_error(f"TIMESYNC: send failed: {exc}")

    def run(self) -> None:
        while not self.stop_event.is_set():
            self._sync_once()
            # stop_event.wait, never time.sleep -- this is why a 30 s sync
            # interval still shuts down in well under 2 s (Fig. 2.6 checklist).
            self.stop_event.wait(self.interval_s)


# ponytail: single-sample offset, no filter. The Pi has no battery RTC, so an
# NTP step mid-flight jumps the offset by one sample interval. Add a 3-sample
# median (lowest-RTT wins) if a flight log ever shows jitter.
