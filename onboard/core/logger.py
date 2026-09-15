"""Logger - a thin adapter over stdlib logging.

The class exists because the class diagram names Logger.log_info/log_error.
It adds exactly one thing of its own: RotatingFileHandler configuration, so a
long flight cannot fill the SD card with the log file instead of images.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class Logger:
    def __init__(self, log_file: Optional[Path] = None, level: int = logging.INFO):
        self._log = logging.getLogger("onboard")
        self._log.setLevel(level)
        self._log.handlers.clear()  # re-init in tests must not stack handlers
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        stream = logging.StreamHandler()  # -> journald under systemd
        stream.setFormatter(fmt)
        self._log.addHandler(stream)

        if log_file is not None:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            rotating = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
            rotating.setFormatter(fmt)
            self._log.addHandler(rotating)

        self.log_info = self._log.info
        self.log_error = self._log.error
        self.log_warning = self._log.warning
