#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial", "numpy", "opencv-python"]
# ///
"""
Live viewer for the ESP32-CAM 54x42 field, read through the RP2040 bridge.

    uv run tools/viewer.py COM8

The dependency block above is PEP 723 metadata, so uv builds the environment on
the first run and nothing has to be installed by hand. With a plain interpreter
instead: pip install pyserial numpy opencv-python, then run it directly.

During bring-up this talks to the bridge port (CDC 0) directly: the RP2040 is
still a transparent pipe, so the ESP32's frames and its text log arrive on the
same stream and commands typed here go straight back to the sketch. Once the
RP2040 grows its own vision stage the same viewer works against CDC 1, which
additionally carries the mask and skeleton layers.

Opening the port is what sets the bridge's UART rate, so --baud must match the
sketch's LINK_BAUD. Nothing else configures it.
"""

import argparse
import binascii
import re
import struct
import sys
import threading
import time

import cv2
import numpy as np
import serial
import serial.tools.list_ports

# Windows 콘솔은 로캘 코드페이지로 인코딩한다(한국어 환경은 cp949). 거기 없는
# 문자가 하나라도 섞이면 print 가 UnicodeEncodeError 를 던져 프로그램을 통째로
# 죽인다 - em-dash 하나 때문에 시작하자마자 죽은 적이 있다. 영문 로캘에서는
# 한글 자체가 그렇게 된다. 글자가 깨지는 편이 죽는 것보다 낫다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass

MAGIC = b"\xA5\x5A"
HDR_LEN = 12
MAX_PAYLOAD = 64 * 1024

TYPE_COVERAGE = 1
TYPE_MASK = 2
TYPE_SKELETON = 3
TYPE_PREVIEW = 4
TYPE_RAW = 6
TYPE_GRAPH = 7          # joints + edges, not an image

IMAGE_TYPES = {TYPE_COVERAGE, TYPE_MASK, TYPE_SKELETON, TYPE_PREVIEW, TYPE_RAW}

TYPE_NAME = {
    TYPE_COVERAGE: "coverage",
    TYPE_MASK: "mask",
    TYPE_SKELETON: "skeleton",
    TYPE_PREVIEW: "preview",
    TYPE_RAW: "raw",
}


def crc16(data):
    """CRC-16/CCITT-FALSE, the same one the firmware computes.

    binascii.crc_hqx is that exact polynomial (0x1021, MSB first, unreflected);
    seeding it with 0xFFFF makes it CCITT-FALSE. It is C, so it does not show up
    in the frame budget the way a Python loop would.
    """
    return binascii.crc_hqx(data, 0xFFFF)


class Link(threading.Thread):
    """Reads the port, splits frames from log text, keeps the latest of each type."""

    daemon = True

    def __init__(self, port, baud):
        super().__init__()
        # Leave DTR/RTS at pyserial's default (both asserted). The bridge's
        # cross-coupling reads that as "release IO0", which is what lets the
        # camera run - IO0 is its XCLK. Poking them one at a time here would
        # briefly make them differ and glitch that clock.
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.frames = {}
        self.stamps = {}
        self.bad_crc = 0
        self.total = 0
        self.running = True

    def run(self):
        while self.running:
            try:
                chunk = self.ser.read(8192)
            except serial.SerialException as exc:
                print(f"\n[serial] {exc}", file=sys.stderr)
                self.running = False
                return
            if chunk:
                self.buf += chunk
                self._parse()

    def stop(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass

    def send(self, line):
        try:
            self.ser.write((line + "\n").encode())
        except serial.SerialException:
            pass

    # -- parsing ---------------------------------------------------------

    def _parse(self):
        buf = self.buf
        while True:
            i = buf.find(MAGIC)
            if i < 0:
                # No header in sight. Everything is log text except a trailing
                # byte that might be the first half of a header.
                keep = 1 if buf[-1:] == b"\xA5" else 0
                if len(buf) > keep:
                    self._text(bytes(buf[: len(buf) - keep]))
                    del buf[: len(buf) - keep]
                return

            if i:
                self._text(bytes(buf[:i]))
                del buf[:i]

            if len(buf) < HDR_LEN:
                return

            typ, _seq, w, h, ln, crc = struct.unpack_from("<BBHHHH", buf, 2)

            known = typ in IMAGE_TYPES or typ == TYPE_GRAPH
            if (not known or ln == 0 or ln > MAX_PAYLOAD
                    or (typ in IMAGE_TYPES and w * h != ln)):
                del buf[:2]          # not a header after all; resync past it
                continue
            if len(buf) < HDR_LEN + ln:
                return

            payload = bytes(buf[HDR_LEN : HDR_LEN + ln])
            if crc16(payload) != crc:
                self.bad_crc += 1
                del buf[:2]
                continue

            del buf[: HDR_LEN + ln]
            value = self._graph(payload) if typ == TYPE_GRAPH else \
                np.frombuffer(payload, np.uint8).reshape(h, w)
            if value is None:
                continue
            with self.lock:
                self.frames[typ] = value
                self.stamps[typ] = time.monotonic()
                self.total += 1

    @staticmethod
    def _graph(p):
        """njoints, nedges, then (x, y, kind) each and (a, b) each."""
        if len(p) < 2:
            return None
        nj, ne = p[0], p[1]
        if len(p) < 2 + nj * 3 + ne * 2:
            return None
        joints = [(p[2 + i * 3], p[3 + i * 3], p[4 + i * 3]) for i in range(nj)]
        base = 2 + nj * 3
        edges = [(p[base + i * 2], p[base + 1 + i * 2]) for i in range(ne)]
        return joints, edges

    def _text(self, raw):
        sys.stdout.write(raw.decode("utf-8", "replace"))
        sys.stdout.flush()

    def snapshot(self):
        with self.lock:
            return dict(self.frames), self.bad_crc, self.total


# -- drawing -------------------------------------------------------------


def panel(img, title, scale, cmap=None, stretch=False):
    """
    stretch rescales the panel's own min..max onto 0..255 before drawing.

    A difference field lives in the bottom fifth of the range - 50 out of 255 is
    a strong reading - so drawn literally it is a black rectangle with a few
    grey specks, and none of the structure that is actually there shows up. The
    title carries the real numbers so the stretch never hides them.
    """
    if stretch:
        lo, hi = int(img.min()), int(img.max())
        if hi > lo:
            img = (((img.astype(np.int32) - lo) * 255) // (hi - lo)).astype(np.uint8)

    big = cv2.resize(
        img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST
    )
    if cmap is not None:
        big = cv2.applyColorMap(big, cmap)
    else:
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(big, (0, 0), (big.shape[1] - 1, big.shape[0] - 1), (60, 60, 60), 1)
    cv2.putText(big, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(big, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return big


def pose_panel(graph, scale, w, h):
    """The stick figure: ends and branches as dots, the runs between as lines."""
    joints, edges = graph
    img = np.zeros((h * scale, w * scale, 3), np.uint8)
    half = scale // 2

    def pt(j):
        return (int(j[0]) * scale + half, int(j[1]) * scale + half)

    for a, b in edges:
        if a < len(joints) and b < len(joints):
            cv2.line(img, pt(joints[a]), pt(joints[b]), (120, 200, 255), 2, cv2.LINE_AA)
    for j in joints:
        # branch points (shoulders, hips) green; ends (head, hands, feet) orange
        colour = (120, 255, 140) if j[2] else (80, 150, 255)
        cv2.circle(img, pt(j), max(3, half - 1), colour, -1, cv2.LINE_AA)

    title = f"pose  {len(joints)} joints  {len(edges)} links"
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (60, 60, 60), 1)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="serial port, e.g. COM7 or /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600, help="must match LINK_BAUD (default 921600)")
    ap.add_argument("--scale", type=int, default=8, help="pixel magnification (default 8)")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    if args.list or not args.port:
        # One composite device, two CDC functions, so both ports carry the same
        # generic description and only the interface number tells them apart.
        # Windows spells that either as MI_nn or as the tail of LOCATION,
        # depending on the driver, so look for both.
        for p in serial.tools.list_ports.comports():
            hwid = (p.hwid or "").upper()
            role = ""
            if "1209:0001" in hwid or "1209&PID_0001" in hwid:
                m = re.search(r"MI_(\d+)", hwid) or re.search(r"LOCATION=\S*[.](\d+)", hwid)
                itf = int(m.group(1)) if m else None
                if itf == 0:
                    role = "   <- ESP32-CAM UART (esptool, close before flashing)"
                elif itf == 2:
                    role = "   <- Vision Stream (give this one to the viewer)"
            print(f"{p.device:10s} {p.description}{role}")
            print(f"{'':10s} {p.hwid}")
        return 0 if args.list else 1

    link = Link(args.port, args.baud)
    link.start()
    print(__doc__)
    print(f"[open] {args.port} @ {args.baud}\n")

    # Mirrors of the sketch's defaults, so the keys can send absolute values.
    # '?' prints what the ESP32 actually holds if these ever drift.
    gain, exposure, sensor_gain = 16, 300, 4
    threshold, opening, fill, skeleton = 40, 1, True, True   # vision.c defaults
    invert = False
    bg_adapt = True
    colormap = True        # difference fields are unreadable in plain grey
    stretch = True
    auto_exposure = False
    preview = False

    fps, fps_n, fps_t0 = 0.0, 0, time.monotonic()
    last_total = 0

    while link.running:
        frames, bad_crc, total = link.snapshot()

        field = frames.get(TYPE_COVERAGE)
        label = "coverage |cur-bg|"
        if field is None:
            field = frames.get(TYPE_RAW)
            label = "raw downscale"

        if field is not None:
            # Turbo runs blue -> green -> yellow -> red, so "how much of this
            # cell changed" reads as a band of colour the way a depth map does.
            lo, hi = int(field.min()), int(field.max())
            panels = [panel(field, f"{label}   {lo}-{hi}", args.scale,
                            cv2.COLORMAP_TURBO if colormap else None, stretch)]

            # Only worth drawing a local preview while the RP2040 is not
            # sending a real mask - during bring-up, before its stage is fed.
            if frames.get(TYPE_MASK) is None and threshold > 0:
                mask = np.where(field >= threshold, 255, 0).astype(np.uint8)
                panels.append(panel(mask, f"local >= {threshold}", args.scale))

            for typ, name in ((TYPE_MASK, "mask"), (TYPE_SKELETON, "skeleton")):
                if frames.get(typ) is not None:
                    panels.append(panel(frames[typ], name, args.scale))

            if frames.get(TYPE_GRAPH) is not None:
                panels.append(pose_panel(frames[TYPE_GRAPH], args.scale,
                                         field.shape[1], field.shape[0]))

            canvas = np.hstack(panels)

            bar = np.zeros((26, canvas.shape[1], 3), np.uint8)
            stats = (f"{fps:5.1f} fps   min {int(field.min()):3d}  "
                     f"mean {int(field.mean()):3d}  max {int(field.max()):3d}   "
                     f"thr {threshold or 'auto'}{' inv' if invert else ''}  open {opening}  "
                     f"bg {'adapt' if bg_adapt else 'FROZEN'}   "
                     f"gain {gain}/16   crc err {bad_crc}")
        # Exposure is deliberately not shown here. The sketch decides it by
        # running the sensor's own loop and freezing the result, so this side
        # does not know the number - and printing a stale guess next to live
        # readings is worse than printing nothing. '/' asks for the real one.
            cv2.putText(bar, stats, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1)
            cv2.imshow("esp32cam", np.vstack([canvas, bar]))

        # What the sensor actually sees, before any of the processing. This is
        # the window that separates an optics problem from a tuning one: if the
        # lens is covered or wildly out of focus it shows up here as a
        # featureless grey, and nothing downstream can be tuned around that.
        prev = frames.get(TYPE_PREVIEW)
        if prev is not None:
            # Always grey: this one is a photograph, and the point of it is to
            # judge framing, focus and exposure. A colour map makes all three
            # unreadable - a flat saturated frame and a well exposed one both
            # come out as a wash of colour.
            img = panel(prev, "camera 160x120", 4)
            bar = np.zeros((26, img.shape[1], 3), np.uint8)
            cv2.putText(bar,
                        f"min {int(prev.min()):3d}  mean {int(prev.mean()):3d}  "
                        f"max {int(prev.max()):3d}   spread {int(prev.max()) - int(prev.min()):3d}"
                        f"   exp {exposure}  sgain {sensor_gain}",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1)
            cv2.imshow("camera", np.vstack([img, bar]))

        # Frame rate of the 54x42 stream, measured over a second.
        now = time.monotonic()
        fps_n += total - last_total
        last_total = total
        if now - fps_t0 >= 1.0:
            fps = fps_n / (now - fps_t0)
            fps_n, fps_t0 = 0, now

        k = cv2.waitKey(16) & 0xFF
        if k == 255:
            continue

        if k in (ord("q"), 27):
            break
        elif k == ord("["):
            threshold = max(0, threshold - 5);   link.send(f"t{threshold}")
        elif k == ord("]"):
            threshold = min(255, threshold + 5); link.send(f"t{threshold}")
        elif k == ord("o"):
            opening = max(0, opening - 1);       link.send(f"o{opening}")
        elif k == ord("O"):
            opening = min(4, opening + 1);       link.send(f"o{opening}")
        elif k == ord("a"):
            stretch = not stretch
        elif k == ord("f"):
            # A fixed camera has no reason to let the model drift, and drift is
            # what dissolves somebody who stops moving.
            bg_adapt = not bg_adapt; link.send(f"a{4 if bg_adapt else 0}")
        elif k == ord("i"):
            invert = not invert;  link.send(f"i{1 if invert else 0}")
        elif k == ord("h"):
            fill = not fill;      link.send(f"h{1 if fill else 0}")
        elif k == ord("k"):
            skeleton = not skeleton; link.send(f"k{1 if skeleton else 0}")
        elif k == ord("c"):
            colormap = not colormap
        elif k == ord("b"):
            link.send("b")          # capture now
        elif k == ord("B"):
            link.send("b8")         # capture in 8 s, time to step out of shot
        elif k == ord("d"):
            link.send("m d")
        elif k == ord("r"):
            link.send("m r")
        elif k == ord("p"):
            # Every 4th frame: one full preview costs ~208 ms of a 921600 link,
            # so sending them back to back would starve the 54x42 stream.
            preview = not preview
            link.send(f"p{4 if preview else 0}")
            if not preview:
                try:
                    cv2.destroyWindow("camera")
                except cv2.error:
                    pass
        elif k == ord("g"):
            gain = max(4, gain - 8);      link.send(f"g{gain}")
        elif k == ord("G"):
            gain = min(248, gain + 8);    link.send(f"g{gain}")
        elif k == ord("e"):
            exposure = max(0, exposure - 50);    link.send(f"e{exposure}")
        elif k == ord("E"):
            exposure = min(1200, exposure + 50); link.send(f"e{exposure}")
        elif k == ord("n"):
            sensor_gain = max(0, sensor_gain - 1);  link.send(f"n{sensor_gain}")
        elif k == ord("N"):
            sensor_gain = min(30, sensor_gain + 1); link.send(f"n{sensor_gain}")
        elif k == ord("x"):
            auto_exposure = not auto_exposure
            link.send(f"x{1 if auto_exposure else 0}")
        elif k == ord("/"):
            link.send("?")
        elif k == ord("s") and field is not None:
            name = time.strftime("field_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, cv2.resize(field, None, fx=args.scale, fy=args.scale,
                                         interpolation=cv2.INTER_NEAREST))
            print(f"\n[saved] {name}")

    link.stop()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
