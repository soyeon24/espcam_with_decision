#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python"]
# ///
"""
Synthetic check for the posture stage. No board and no camera needed.

    uv run tools/test_posture.py

Builds 54x42 masks by hand, pushes them through esp_source.mask_to_zone and the
tracker, and asserts the label that comes out - the same path a real frame from
the bridge takes. Also checks the wire framing, so a change to either end of
esp_source is covered.

Exits non-zero on the first failure.
"""
from __future__ import annotations

import binascii
import collections
import contextlib
import io
import struct
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import esp_source as es
from posture import PostureTracker, ZONE_COLS as W, ZONE_ROWS as H

FPS = 30.0
Y, X = np.mgrid[0:H, 0:W]
failed = 0


def person(head_row=6.0, scale=1.0, hand=None, body=None):
    """A head and a torso. scale < 1 is the same person further away.

    body scales the torso alone. A nod moves the head without moving the torso
    much closer or further, so the two have to be separable to model one: the
    recorded runs show head width holding or growing slightly through nodding
    while the silhouette below it shifts around.
    """
    cx = W / 2
    body = scale if body is None else body
    head_r = 5.0 * scale
    m = ((X - cx) ** 2 / head_r ** 2 + (Y - (head_row + 4 * scale)) ** 2 / head_r ** 2) <= 1.0
    m |= (((X - cx) ** 2 / (13.0 * body) ** 2
           + (Y - (head_row + 20 * body)) ** 2 / (15.0 * body) ** 2) <= 1.0)
    if hand:
        col = cx if hand == "over" else cx + 16
        m |= (np.abs(X - col) <= 1) & (Y <= head_row + 22)
    return m.astype(np.uint8)


def nod(t, period=6.0, rest=6.0, depth=9.0):
    """Head sinks over 1.5 s, snaps back over 0.5 s, then rests."""
    phase = t % period
    if phase < 1.5:
        return rest + depth * (phase / 1.5)
    if phase < 2.0:
        return rest + depth * (1.0 - (phase - 1.5) / 0.5)
    return rest


def calibrated():
    """A tracker past both calibration steps, and the clock it left off at."""
    tr, t = PostureTracker(), 0.0
    empty = np.zeros((H, W), np.uint8)
    tr.start_background(20)
    for _ in range(20):
        t += 1 / FPS
        tr.update(es.mask_to_zone(empty), t)
    tr.start_baseline(40)
    for _ in range(40):
        t += 1 / FPS
        tr.update(es.mask_to_zone(person()), t)
    assert tr.baseline is not None, "baseline was not captured"
    return tr, t


def hold(tr, t, pose_fn, seconds):
    for i in range(int(seconds * FPS)):
        t += 1 / FPS
        v = tr.update(es.mask_to_zone(pose_fn(i / FPS)), t)
    return v, t


# -- the wire format ------------------------------------------------------
link = object.__new__(es._Link)
link.buf, link.frames, link.stamps, link.lock = bytearray(), {}, {}, threading.Lock()
link.log = collections.deque(maxlen=8)      # _text 가 남기는 ESP 로그 줄
payload = bytes((i * 7) % 256 for i in range(W * H))
good = struct.pack("<BBHHHH", es.TYPE_MASK, 0, W, H, len(payload),
                   binascii.crc_hqx(payload, 0xFFFF))
link.buf += b"log text\n" + es.MAGIC + good + payload
with contextlib.redirect_stdout(io.StringIO()):     # the parser echoes log text
    link._parse()
got = link.frames.get(es.TYPE_MASK)
ok = got is not None and got.shape == (H, W) and bytes(got.ravel()) == payload
failed += not ok
print(f"frame parse      {'ok' if ok else 'FAIL'}")

link.frames.clear(), link.buf.clear()
bad = struct.pack("<BBHHHH", es.TYPE_MASK, 0, W, H, len(payload), 0x0000)
link.buf += es.MAGIC + bad + payload
with contextlib.redirect_stdout(io.StringIO()):
    link._parse()
ok = es.TYPE_MASK not in link.frames
failed += not ok
print(f"bad crc dropped  {'ok' if ok else 'FAIL'}\n")

# -- the judgement --------------------------------------------------------
# A slump has to be held: SLUMP_HOLD_S is what separates it from the bottom of
# a nod, where the geometry is identical.
CASES = [
    ("upright",       lambda s: person(),                          5.0, "UPRIGHT"),
    ("slump",         lambda s: person(head_row=19.0),             5.0, "SLUMP"),
    # scale 0.60 puts the reconstructed distance +77% above the reference, which
    # is where a recorded recline sits (+53 to +86%). 0.70 only reached +35% and
    # was asking the judgement to call something the sensor never sees.
    ("recline",       lambda s: person(head_row=9.0, scale=0.60),  5.0, "RECLINE"),
    ("drowsy",        lambda s: person(head_row=nod(s)),          40.0, "DROWSY"),
    ("absent",        lambda s: np.zeros((H, W), np.uint8),        2.0, "ABSENT"),
    ("hand to side",  lambda s: person(hand="side"),               5.0, "UPRIGHT"),
    ("hand overhead", lambda s: person(hand="over"),               5.0, "UPRIGHT"),
    ("brief dip",     lambda s: person(head_row=19.0),             2.0, "UPRIGHT"),
]

print(f"{'case':16s} {'got':9s} {'want':9s} {'head':>5s} {'dist':>8s} "
      f"{'slump':>6s} {'recl':>5s} {'drow':>5s}")
for name, pose, seconds, want in CASES:
    tr, t = calibrated()
    v, t = hold(tr, t, pose, seconds)
    ok = v.label == want
    failed += not ok
    p = v.parts
    print(f"{name:16s} {v.label:9s} {want:9s} {v.features.top_row:5.3f} "
          f"{p.get('dist_mm', 0):+7.0f}mm {p.get('slump', 0):6.2f} "
          f"{p.get('recline', 0):5.2f} {p.get('drowsy', 0):5.2f}"
          f"{'' if ok else '   <-- FAIL'}")

# A nod takes the upper body with it, so the silhouette below the head shifts
# while the head itself stays where it is. Direction is read from head width
# precisely so that shift cannot be mistaken for leaning away - and recline sits
# first in the label order, so if it ever were, it would take the label off
# somebody falling asleep. Every frame has to be checked, not just the last one:
# the disturbance only exists while the head is down.
tr, t = calibrated()
seen, peak = collections.Counter(), 0.0
for i in range(int(40 * FPS)):
    t += 1 / FPS
    s = i / FPS
    v = tr.update(es.mask_to_zone(
        person(head_row=nod(s), body=0.72 if (s % 6.0) < 2.0 else 1.0)), t)
    seen[v.label] += 1
    peak = max(peak, v.parts.get("recline", 0.0))
print()
print(f"nod with body sway: peak recline {peak:.2f}, "
      f"{seen['RECLINE']}/{sum(seen.values())} frames read as RECLINE")
if seen["RECLINE"] or not seen["DROWSY"]:
    failed += 1
    print("   <-- FAIL: a nod must not read as reclining")

# Waking up has to clear quickly; the 30 s counting window would otherwise hold
# DROWSY on screen long after the nodding stopped.
tr, t = calibrated()
v, t = hold(tr, t, lambda s: person(head_row=nod(s)), 30.0)
assert v.label == "DROWSY", f"expected DROWSY before the recovery test, got {v.label}"
back, t1 = None, t
for _ in range(int(40 * FPS)):
    t += 1 / FPS
    v = tr.update(es.mask_to_zone(person()), t)
    if back is None and v.label == "UPRIGHT":
        back = t - t1
print(f"\nwake-up recovery: {f'{back:.1f} s' if back else 'never'}")
if back is None or back > 20.0:
    failed += 1
    print("   <-- FAIL: should return to UPRIGHT within 20 s")

print(f"\n{'all passed' if not failed else f'{failed} failed'}")
sys.exit(1 if failed else 0)
