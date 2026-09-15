"""MavlinkClient -- autopilot link and the single MAVLink reader.

This is the ONLY thread in the process that calls ``recv_match()``. mavutil
connections are not thread safe; a second reader (ClockSync polling for its own
TIMESYNC reply, in the original plan) steals GLOBAL_POSITION_INT from this loop
and loses its own replies into it. Everything that needs a message gets it
dispatched from here instead.

pymavlink is imported under a guard so the module -- and the test suite --
load on a machine that has never had it installed.
"""

import threading
import time

try:
    from pymavlink import mavutil
except ImportError:  # dev box, or a Pi before `pip install pymavlink`
    mavutil = None


class MavlinkClient:
    def __init__(self, url: str, baud: int = 57600, telemetry_hz: int = 5, logger=None):
        self.url = url
        self.baud = baud
        self.telemetry_hz = telemetry_hz
        self.logger = logger
        self.conn = None
        self._send_lock = threading.Lock()  # guards sends only; one reader needs no lock

    # --- link -------------------------------------------------------------

    def connect(self, timeout: float = 10.0) -> None:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed (pip install pymavlink)")
        self.conn = mavutil.mavlink_connection(self.url, baud=self.baud)
        if self.conn.wait_heartbeat(timeout=timeout) is None:
            raise RuntimeError(f"no MAVLink heartbeat from {self.url} within {timeout}s")
        if self.logger:
            self.logger.log_info("mavlink: heartbeat from system %d component %d"
                                 % (self.conn.target_system, self.conn.target_component))
        self.request_streams(self.telemetry_hz)

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def request_streams(self, hz: int) -> None:
        """Ask for position and system time.

        Both the legacy stream request and SET_MESSAGE_INTERVAL are sent:
        ArduPilot honours the first, PX4 the second, and neither errors on the
        one it ignores.
        """
        interval_us = int(1e6 / max(hz, 1))
        with self._send_lock:
            self.conn.mav.request_data_stream_send(
                self.conn.target_system, self.conn.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION, hz, 1)
            for msg_id in (mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                           mavutil.mavlink.MAVLINK_MSG_ID_SYSTEM_TIME):
                self.conn.mav.command_long_send(
                    self.conn.target_system, self.conn.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                    msg_id, interval_us, 0, 0, 0, 0, 0)

    def send_timesync(self, tc1: int, ts1: int) -> None:
        """Send a TIMESYNC request. Passed to ClockSync as a plain callable."""
        with self._send_lock:
            try:
                # MAVLink v2 added target_system/target_component as extension
                # fields, which pymavlink generates as required positional args.
                self.conn.mav.timesync_send(tc1, ts1, self.conn.target_system,
                                            self.conn.target_component)
            except TypeError:
                self.conn.mav.timesync_send(tc1, ts1)  # older 2-arg dialect

    # --- the one reader ---------------------------------------------------

    def run(self) -> None:
        from .types import Telemetry

        while not self.stop_event.is_set():
            try:
                msg = self.conn.recv_match(
                    type=["GLOBAL_POSITION_INT", "SYSTEM_TIME", "TIMESYNC"],
                    blocking=True, timeout=1.0)
            except Exception as exc:
                self.logger.log_error(f"mavlink: receive failed: {exc}")
                self.stop_event.wait(1.0)  # a flapping link must not spin a core
                continue
            if msg is None:
                continue  # 1 s timeout is what keeps this loop stoppable

            now = time.time()
            kind = msg.get_type()
            if kind == "TIMESYNC":
                self.clock_sync.handle_timesync(msg, time.time_ns())
            elif kind == "SYSTEM_TIME":
                self.clock_sync.set_utc_offset(msg.time_unix_usec / 1e6, now)
            elif kind == "GLOBAL_POSITION_INT":
                sample = Telemetry(
                    t_boot_s=msg.time_boot_ms / 1000.0,
                    lat=msg.lat / 1e7, lon=msg.lon / 1e7,
                    alt_msl_m=msg.alt / 1000.0, rel_alt_m=msg.relative_alt / 1000.0,
                    heading_deg=None if msg.hdg == 65535 else msg.hdg / 100.0)
                self.clock_sync.bootstrap(sample.t_boot_s, now)
                with self.telemetry_lock:
                    self.telemetry_buffer.append(sample)

        self.logger.log_info("mavlink: reader stopped")

    def attach(self, telemetry_buffer, telemetry_lock, clock_sync, stop_event) -> "MavlinkClient":
        """Wire up the dispatch targets. Separate from __init__ so the link can
        be opened in parallel with the camera before the queues exist."""
        self.telemetry_buffer = telemetry_buffer
        self.telemetry_lock = telemetry_lock
        self.clock_sync = clock_sync
        self.stop_event = stop_event
        return self
