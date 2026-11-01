"""
Keyboard Controls for Experiments

Space pauses and continues; further keys can trigger callbacks (e.g. disturbances). The arrow keys are named "up",
"down", "left" and "right"; a callback without a description is neither listed nor announced when its key is pressed.

    keys = KeyboardControls(paused=True, callbacks={"d": ("drop the cube", drop)})
    for ...:
        paused_for = keys.wait()   # blocks while paused, returns the seconds spent paused
        ...
    keys.close()

Reads single key presses from the terminal (POSIX only); without an interactive terminal it does nothing.

Authors: Adrian Pfisterer
License: MIT
"""

__author__ = "Adrian Pfisterer"
__copyright__ = "Copyright 2025 RBO TU Berlin"
__license__ = "MIT"
__version__ = "1.0.0"

import atexit
import sys
import threading
import time
from typing import Callable, Dict, Optional, Sequence, Tuple

# the last character of the escape sequences the terminal sends for the arrow keys
ARROW_KEYS = {"A": "up", "B": "down", "C": "right", "D": "left"}


class KeyboardControls:
    """Space toggles between running and paused; the keys in `callbacks` ({key: (description, function)}) call their
    function. `hints` are listed with the keys (e.g. for keys whose callbacks have no description)."""

    def __init__(
        self,
        paused: bool = False,
        callbacks: Dict[str, Tuple[Optional[str], Callable]] = None,
        hints: Sequence[str] = (),
        verbose: bool = True,
    ):
        self.callbacks = callbacks or {}
        self.verbose = verbose
        self.paused = False
        self._settings = None
        try:
            import termios
            import tty

            if not sys.stdin.isatty():
                return
            fd = sys.stdin.fileno()
            self._settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)  # key presses arrive without Enter, Ctrl+C still works
        except (ImportError, OSError):
            return
        atexit.register(self.close)
        self.paused = paused
        threading.Thread(target=self._read_keys, daemon=True).start()
        if verbose:
            help_text = ", ".join(
                [f"{k.upper()}: {description}" for k, (description, _) in self.callbacks.items() if description]
                + list(hints)
            )
            print(
                ("Paused, press space to start" if paused else "Space: pause") + (f"; {help_text}" if help_text else "")
            )

    def _read_keys(self):
        while self._settings is not None:
            key = sys.stdin.read(1)
            if key == "\x1b":  # escape sequence, e.g. ESC [ A for the up arrow
                if sys.stdin.read(1) != "[":
                    continue
                key = ARROW_KEYS.get(sys.stdin.read(1), "")
            key = key.lower()
            if key == " ":
                self.paused = not self.paused
                if self.verbose:
                    print("Paused, press space to continue" if self.paused else "Running")
            elif key in self.callbacks:
                description, function = self.callbacks[key]
                if self.verbose and description:
                    print(description[:1].upper() + description[1:])
                function()

    def wait(self, while_paused=None) -> float:
        """Block while paused (calling `while_paused`, e.g. to keep a viewer responsive); returns the paused time."""
        start = time.monotonic()
        while self.paused:
            if while_paused is not None:
                while_paused()
            time.sleep(0.02)
        return time.monotonic() - start

    def close(self):
        """Restore the terminal."""
        if self._settings is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._settings)
            self._settings = None
        self.paused = False
