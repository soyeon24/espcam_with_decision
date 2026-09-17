#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial", "numpy", "opencv-python"]
# ///
"""
Just the camera. One window, one picture, no controls.

    uv run tools/camview.py COM5

viewer.py answers "is the pipeline doing the right thing" and has a key for
every stage to prove it. This answers only "what is the camera looking at",
which is the question worth separating out: framing, focus and exposure are
decided at the camera and everything downstream is powerless if they are wrong.

It turns the preview stream on at startup and off again on the way out, so it
leaves the link the way it found it. Frame parsing comes from viewer.py rather
than a second copy of it.
"""

import argparse
import sys
import time

import cv2
import numpy as np
import serial.tools.list_ports

from viewer import Link, TYPE_PREVIEW

# Windows 콘솔은 로캘 코드페이지로 인코딩한다(한국어 환경은 cp949). 거기 없는
# 문자가 하나라도 섞이면 print 가 UnicodeEncodeError 를 던져 프로그램을 통째로
# 죽인다 - em-dash 하나 때문에 시작하자마자 죽은 적이 있다. 영문 로캘에서는
# 한글 자체가 그렇게 된다. 글자가 깨지는 편이 죽는 것보다 낫다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="the Vision Stream port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--scale", type=int, default=5, help="magnification (default 5)")
    args = ap.parse_args()

    if not args.port:
        for p in serial.tools.list_ports.comports():
            print(f"{p.device:10s} {p.description}")
        return 1

    link = Link(args.port, args.baud)
    link.start()
    link.send("p1")                 # every frame; nothing else is being shown
    print(f"[open] {args.port} - q or ESC to quit\n")

    waited = time.monotonic()
    try:
        while link.running:
            frames, _bad, _total = link.snapshot()
            img = frames.get(TYPE_PREVIEW)

            if img is None:
                if time.monotonic() - waited > 4:
                    print("[wait] no camera frames yet - is this the Vision Stream port?")
                    waited = time.monotonic()
                if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                    break
                continue

            big = cv2.resize(img, (img.shape[1] * args.scale, img.shape[0] * args.scale),
                             interpolation=cv2.INTER_NEAREST)
            big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

            lo, hi, mean = int(img.min()), int(img.max()), int(img.mean())
            bar = np.zeros((26, big.shape[1], 3), np.uint8)
            # spread is the number that says whether the picture is usable at
            # all: a covered lens and a blown-out frame are both flat, and both
            # look like a broken camera from any later stage.
            cv2.putText(bar, f"min {lo:3d}   mean {mean:3d}   max {hi:3d}   spread {hi - lo:3d}",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1)
            cv2.imshow("camera", np.vstack([big, bar]))

            if cv2.waitKey(16) & 0xFF in (ord("q"), 27):
                break
    finally:
        link.send("p0")
        time.sleep(0.1)             # let it go out before the port closes
        link.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
