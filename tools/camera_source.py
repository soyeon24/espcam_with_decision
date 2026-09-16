"""
A webcam standing in for the ESP32-CAM and the RP2040, so the posture stage can
be run and tuned without either board.

It is a port of what the two of them already do, in the same order and with the
same knobs, so what comes out is the thing the hardware would have sent:

    grey frame -> sigma-delta background -> |cur - bg| -> box down to 54x42
                                                              |
                  largest blob <- open <- fill holes <- threshold
                       |
                    mask + coverage

esp32cam_sender.ino owns the first row, vision.c the second. Numbers that differ
from the firmware defaults are called out where they are set.

This is a stand-in, not a simulator: a webcam's auto-exposure, its colour
processing and its resolution are all different, so thresholds tuned here still
want checking on the board. What it does reproduce faithfully is the *structure*
of the signal - in particular, that a brightness difference only marks a subject
where the subject differs in brightness, which is the interesting failure.
"""
from __future__ import annotations

import cv2
import numpy as np

from esp_source import mask_to_zone
from posture import ZoneFrame

W, H = 54, 42

SRC_W, SRC_H = 160, 120     # QQVGA, as the sketch captures
BG_PERIOD = 4               # sigma-delta step every N frames (sketch: 4)
BG_GUARD = 24               # do not absorb pixels this far off (sketch: 24)
GAIN_Q4 = 16                # difference gain, 16 = 1.0 (sketch: 16)
SETTLE_FRAMES = 20          # discard this many before seeding the model


def _erode(m):
    return cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)


def _dilate(m):
    return cv2.dilate(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REPLICATE)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood the background in from the border; promote whatever it never reached.

    vision.c uses this instead of a morphological closing because at 54x42 a
    person's legs are one cell apart and a single dilation fuses them. The flood
    is 4-connected on purpose, so a diagonal chain of foreground counts as a wall.
    """
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask
    outside = np.zeros((h + 4, w + 4), np.uint8)
    flooded = pad.copy()
    cv2.floodFill(flooded, outside, (0, 0), 1, flags=4)
    return (mask | (flooded[1:-1, 1:-1] == 0).astype(np.uint8)).astype(np.uint8)


def largest_blob(mask: np.ndarray) -> np.ndarray:
    """Keep the biggest connected component; a difference field speckles the rest."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


class CameraZoneSource:
    """Opens a webcam and produces what EspZoneSource would have.

    Same interface as esp_source.EspZoneSource - read() gives a ZoneFrame - so
    posture_viewer and posture.py cannot tell which one they are attached to.
    Distances are estimated from apparent size, exactly as the ESP path does,
    because neither has a depth sensor behind it.
    """

    def __init__(self, camera: int = 0, threshold: int = 40, opening: int = 1,
                 fill: bool = True, largest: bool = True):
        self.cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.threshold = threshold
        self.opening = opening
        self.fill = fill
        self.largest = largest

        self._bg: np.ndarray | None = None
        self._settle = SETTLE_FRAMES
        self._n = 0
        self.coverage: np.ndarray | None = None   # for display only
        self.stale = False

    def opened(self) -> bool:
        return self.cap.isOpened()

    def reset_background(self) -> None:
        """Reseed from the next frame. The board calls this 'b'."""
        self._bg = None
        self._settle = SETTLE_FRAMES

    def read(self) -> ZoneFrame | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grey = cv2.resize(grey, (SRC_W, SRC_H), interpolation=cv2.INTER_AREA)

        self._n += 1
        if self._settle > 0:
            self._settle -= 1
            self.coverage = np.zeros((H, W), np.uint8)
            return mask_to_zone(np.zeros((H, W), np.uint8))
        if self._bg is None:
            self._bg = grey.astype(np.int16)
            self.coverage = np.zeros((H, W), np.uint8)
            return mask_to_zone(np.zeros((H, W), np.uint8))

        diff = grey.astype(np.int16) - self._bg
        # Sigma-delta: one level per step, and never towards a pixel that
        # currently reads as foreground - otherwise somebody sitting still
        # dissolves into the background in a few seconds.
        if self._n % BG_PERIOD == 0:
            movable = np.abs(diff) <= BG_GUARD
            self._bg += np.sign(diff) * movable

        # The absolute value is taken per source pixel, before the box average.
        # box(|cur-bg|) is not |box(cur)-box(bg)|, and the latter cancels wherever
        # a cell holds both a brighter and a darker part of the subject - which is
        # most of an outline.
        absdiff = np.abs(diff).astype(np.float32)
        coverage = cv2.resize(absdiff, (W, H), interpolation=cv2.INTER_AREA)
        coverage = np.clip(coverage * GAIN_Q4 / 16.0, 0, 255).astype(np.uint8)

        mask = (coverage >= self.threshold).astype(np.uint8)
        if self.fill:
            mask = fill_holes(mask)          # before opening: an erosion opens holes
        for _ in range(self.opening):        # to the border, and then nothing fills them
            mask = _erode(mask)
        for _ in range(self.opening):
            mask = _dilate(mask)
        if self.largest and mask.any():
            mask = largest_blob(mask)
        self.coverage = coverage
        return mask_to_zone(mask)

    def send(self, line: str) -> None:
        """Accept the board's commands so the viewer needs no special case."""
        if line.startswith("t") and line[1:].isdigit():
            self.threshold = int(line[1:])
        elif line.startswith("b"):
            # 'b8' is 'capture in 8 s' on the board. The viewer runs that
            # countdown itself so both paths behave the same, and by the time
            # this arrives the delay is already over.
            self.reset_background()

    def log_lines(self) -> list[str]:
        """No board, so nothing to report. Present so the viewer needs no branch."""
        return []

    @property
    def error(self) -> str | None:
        return None

    def release(self) -> None:
        self.cap.release()
