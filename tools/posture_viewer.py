#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyserial", "numpy", "opencv-python"]
# ///
"""54x42 필드와 자세 판정을 한 화면에 띄운다. 판정까지 하는 도구는 이것 하나다.

viewer.py 는 파이프라인을 눈으로 튜닝하는 도구라 mask·skeleton 을 원본 그대로
흑백으로 그리고 판정은 하지 않는다. 이 파일이 그 튜닝 키를 그대로 들고 오면서
열화상 표시와 자세 판정을 함께 한다.

왼쪽: 센서가 보내는 coverage 필드를 열화상 팔레트로 그린 것(배경과 다를수록 뜨겁게).
      v 를 누르면 판정이 실제로 쓰는 zone 거리 배열로 바꿔 볼 수 있다.
오른쪽: 현재 자세 라벨과 FSM 으로 나갈 phi/delta, 그리고 판정 근거가 된 기하 특징.

진행 순서
  1) 배경 캘리브레이션 - SPACE. 자리를 비울 8초를 센 뒤 센서에 배경 재캡처를
     시키고(ESP 는 재노출까지 다시 한다), 그게 끝난 뒤에 zone 기준 거리를 잡는다.
  2) 자세 baseline    - 바른 자세로 앉아서 SPACE. 이후 판정은 이 기준 대비 상대값이다.
  3) 판정             - UPRIGHT / SLUMP(엎드림) / RECLINE(젖힘) / DROWSY(졸음) / ABSENT

판정은 zone 배열만 본다. 센서는 read() -> ZoneFrame 을 내놓기만 하면 되므로,
웹캠 스텁이든 ESP32-CAM 이든 실 ToF(VL53L9CX)든 이 파일은 그대로다.

화면에 찍는 글자는 전부 ASCII 다 - cv2.putText 는 한글을 그리지 못한다.

Keys:
  단계   SPACE 다음 단계 (1단계 카운트다운 중 다시 누르면 즉시 캡처)
         n 배경 다시      b 자세 baseline 다시
         1/2/3/4 실측 라벨 기록 (upright/slump/recline/drowsy)   0 기록 정지
  화면   v 레이어 (coverage / zone / mask / skeleton)   c 팔레트
         a 대비 스트레치   g 격자   s 스냅샷 저장   p 카메라 사진 창
  보드   [ ] 임계값   o O opening   h 홀 필링   i invert
         - = 차분 게인   , . 노출   x 자동노출   / 상태 출력
         f 배경 적응 on/off   w W 흡수 가드 (잔상이 남으면 낮춰 볼 것)
  q / ESC  종료
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from posture import ZONE_COLS, ZONE_ROWS, PostureTracker, PostureVerdict
from palette import PALETTES, sensor_grid

# Windows 콘솔은 로캘 코드페이지로 인코딩한다(한국어 환경은 cp949). 거기 없는
# 문자가 하나라도 섞이면 print 가 UnicodeEncodeError 를 던져 프로그램을 통째로
# 죽인다 - em-dash 하나 때문에 시작하자마자 죽은 적이 있다. 영문 로캘에서는
# 한글 자체가 그렇게 된다. 글자가 깨지는 편이 죽는 것보다 낫다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass

LABEL_COLOR = {
    "UPRIGHT": (110, 220, 110),
    "SLUMP": (70, 70, 245),
    "DROWSY": (200, 120, 255),
    "RECLINE": (60, 210, 245),
    "ABSENT": (150, 150, 150),
    "BASELINE": (240, 200, 90),
    "UNKNOWN": (180, 180, 180),
}
PANEL_W = 340
PANEL_MIN_H = 705    # 패널 내용이 다 들어가는 최소 높이. 더 짧으면 푸터가 겹친다.
TAG_KEYS = {ord("1"): "upright", ord("2"): "slump", ord("3"): "recline",
            ord("4"): "drowsy"}
NEAR_MM, FAR_MM = 450.0, 2600.0

# 왼쪽에 그릴 수 있는 것들. zone 은 항상 있고(판정이 쓰는 배열이다), 나머지는
# 센서가 보내줄 때만 있다 - 웹캠 스텁에는 skeleton 이 없다.
LAYERS = ("coverage", "zone", "mask", "skeleton")

STEP_BACKGROUND, STEP_BASELINE, STEP_LIVE = range(3)
STEP_PROMPT = {
    STEP_BACKGROUND: ("STEP 1  background", "Press SPACE, then step out of shot"),
    STEP_BASELINE: ("STEP 2  posture baseline", "Sit upright, then press SPACE"),
    STEP_LIVE: ("", ""),
}

# 1단계는 두 겹이다. 센서 자신의 배경(ESP 는 재노출까지)을 먼저 새로 잡고, 그 다음에
# zone 기준 거리를 뜬다. 순서를 지키지 않으면 사람이 섞인 배경 위에 기준이 굳는다.
BG_CLEAR_S = 8.0          # 화면 밖으로 나갈 시간
BG_SETTLE_MIN_S = 1.5     # 'b' 를 보낸 뒤 최소 대기
BG_SETTLE_MAX_S = 10.0    # 완료 신호가 없어도 이만큼이면 넘어간다
BG_DONE_MARK = "bg captured"   # ESP 가 끝났다고 알리는 문구

# 배경이 오염되면 화면 전체가 '사람'으로 잡힌다. 책상 앞 사람은 zone 의 절반을
# 넘기지 않으므로, 이만큼 차 있으면서 움직이지도 않으면 사람이 아니라 배경이 틀린
# 것이다. 실측: 배경이 깨졌을 때 점유율 64.9%, 시간축 표준편차 0.8.
# 이 상태에서도 판정은 태연히 UPRIGHT 를 뱉으므로, 화면에 말해주지 않으면 모른다.
BG_SUSPECT_OCC = 0.55
BG_SUSPECT_MOTION = 0.010


def render_zones(depth_mm: np.ndarray, palette: int, cell: int, grid: bool) -> np.ndarray:
    """zone 거리 배열 -> 열화상 이미지. 가까울수록 뜨겁다."""
    d = np.where(np.isfinite(depth_mm), depth_mm, FAR_MM)
    heat = 1.0 - np.clip((d - NEAR_MM) / (FAR_MM - NEAR_MM), 0.0, 1.0)
    u8 = (heat * 255.0).astype(np.uint8)

    _, cmap = PALETTES[palette]
    small = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if cmap is None else cv2.applyColorMap(u8, cmap)
    out = cv2.resize(small, (ZONE_COLS * cell, ZONE_ROWS * cell), interpolation=cv2.INTER_NEAREST)
    if grid:
        sensor_grid(out, cell)
    return out


def render_coverage(cov: np.ndarray, palette: int, cell: int, grid: bool,
                    stretch: bool) -> np.ndarray:
    """coverage 필드 -> 열화상 이미지. 배경과 다를수록 뜨겁다.

    이건 거리가 아니라 '이 칸이 배경과 얼마나 다른가'다. 그래서 zone 거리 배열과
    달리 NEAR/FAR 매핑을 쓰지 않는다.

    차이값은 범위의 아래쪽에 몰려 있다 - 255 중 50이면 이미 강한 값이다. 그대로
    그리면 거의 검은 사각형에 점 몇 개가 되어 구조가 안 보이므로 기본으로 자기
    min..max 를 0..255 로 편다. 판정에 들어가는 값은 건드리지 않는다, 화면만 편다.
    """
    u8 = cov
    if stretch:
        lo, hi = int(cov.min()), int(cov.max())
        if hi > lo:
            u8 = (((cov.astype(np.int32) - lo) * 255) // (hi - lo)).astype(np.uint8)

    _, cmap = PALETTES[palette]
    small = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR) if cmap is None else cv2.applyColorMap(u8, cmap)
    out = cv2.resize(small, (cov.shape[1] * cell, cov.shape[0] * cell),
                     interpolation=cv2.INTER_NEAREST)
    if grid:
        sensor_grid(out, cell)
    return out


def _text(img, s, org, scale=0.5, color=(235, 235, 235), weight=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), weight + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _bar(img, org, width, value, color):
    x, y = org
    cv2.rectangle(img, (x, y), (x + width, y + 11), (55, 55, 60), -1)
    filled = int(width * float(np.clip(value, 0.0, 1.0)))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + 11), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + 11), (95, 95, 100), 1)


def render_plain(img: np.ndarray, cell: int, grid: bool) -> np.ndarray:
    """mask·skeleton 처럼 0/255 뿐인 레이어. 팔레트를 씌워도 2색이라 의미가 없으므로
    viewer.py 와 같이 흑백 그대로 그린다."""
    out = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                     (img.shape[1] * cell, img.shape[0] * cell),
                     interpolation=cv2.INTER_NEAREST)
    if grid:
        sensor_grid(out, cell)
    return out


def render_panel(verdict: PostureVerdict, height: int, *, step: int,
                 bg_ready: bool, base_ready: bool, fps: float,
                 link: tuple[bool, str] | None = None,
                 step_hint: str | None = None) -> np.ndarray:
    p = np.full((height, PANEL_W, 3), 26, np.uint8)
    f = verdict.features
    color = LABEL_COLOR.get(verdict.label, (200, 200, 200))
    bar_w = PANEL_W - 32

    title, hint = STEP_PROMPT[step]
    if step_hint:
        hint = step_hint
    if title:
        _text(p, title, (16, 26), 0.5, (240, 200, 90))
        _text(p, hint, (16, 46), 0.42, (200, 200, 205))
        y0 = 74
    else:
        y0 = 30

    _text(p, "POSTURE", (16, y0), 0.45, (150, 150, 155))
    _text(p, verdict.label, (16, y0 + 40), 1.1, color, 2)
    if verdict.note:
        _text(p, verdict.note, (16, y0 + 62), 0.42, (165, 165, 170))
    _text(p, f"present  {'yes' if verdict.present else 'no'}", (16, y0 + 84), 0.44,
          (110, 220, 110) if verdict.present else (150, 150, 150))

    y = y0 + 116
    _text(p, "-> FSM posture Signal", (16, y), 0.44, (150, 150, 155))
    for key, val, col in (("phi   (focus)", verdict.phi, (200, 190, 110)),
                          ("delta (fatigue)", verdict.delta, (90, 130, 245))):
        y += 24
        _text(p, key, (16, y), 0.44)
        _text(p, f"{val:.2f}", (PANEL_W - 58, y), 0.44, col)
        _bar(p, (16, y + 6), bar_w, val, col)
        y += 18

    if verdict.parts:
        y += 22
        _text(p, "deviation", (16, y), 0.44, (150, 150, 155))
        for key, col in (("slump", (70, 70, 245)), ("recline", (60, 210, 245)),
                         ("drowsy", (200, 120, 255))):
            val = verdict.parts.get(key, 0.0)
            y += 24
            _text(p, key, (16, y), 0.44)
            _text(p, f"{val:.2f}", (PANEL_W - 58, y), 0.44, col)
            _bar(p, (16, y + 6), bar_w, val, col)
            y += 18

    y += 22
    _text(p, "zone geometry", (16, y), 0.44, (150, 150, 155))
    head = "--" if not np.isfinite(f.head_mm) else f"{f.head_mm:.0f} mm"
    # dist vs base 가 엎드림/젖힘을 가르는 유일한 축이라 눈에 띄게 둔다.
    dd = verdict.parts.get("dist_mm")
    dist_txt = "--" if dd is None else f"{dd:+.0f} mm"
    for key, val in (("occupancy", f"{f.occupancy * 100:.1f} %"),
                     ("head row", f"{f.top_row:.3f}"),
                     ("spread", f"{f.spread:.3f}"),
                     ("head dist", head),
                     ("dist vs base", dist_txt),
                     ("motion", f"{f.motion * 100:.1f} %"),
                     ("nod / min", f"{verdict.nod_rate:.1f}"),
                     # 꾸벅임이 문턱에 얼마나 모자랐는지. 왼쪽이 이번 하강 깊이,
                     # 오른쪽이 인정에 필요한 깊이다.
                     ("nod dip/need", f"{verdict.parts.get('nod_dip', 0.0):.3f} / "
                                      f"{verdict.parts.get('nod_amp', 0.0):.3f}")):
        y += 21
        _text(p, key, (16, y), 0.43, (175, 175, 180))
        _text(p, val, (PANEL_W - 122, y), 0.43)

    if link is not None:
        ok, note = link
        # 링크가 멈춰도 마지막 프레임이 계속 그려져 화면은 멀쩡해 보인다.
        # 그 차이를 눈으로 알 수 있는 유일한 줄이다.
        _text(p, note[:46], (16, height - 72), 0.4,
              (110, 220, 110) if ok else (70, 70, 245))
    _text(p, f"background {'captured' if bg_ready else 'auto-learning'}", (16, height - 54), 0.43,
          (110, 220, 110) if bg_ready else (60, 210, 245))
    _text(p, f"baseline   {'ok' if base_ready else 'not set'}", (16, height - 36), 0.43,
          (110, 220, 110) if base_ready else (60, 210, 245))
    _text(p, f"{fps:4.1f} fps  SPACE next  n bg  b base  v layer  q quit",
          (16, height - 14), 0.4, (140, 140, 145))
    return p


def compose(depth_mm, verdict, *, step=STEP_LIVE, palette=0, cell=14, grid=False,
            bg_ready=False, base_ready=False, fps=0.0, layers=None, layer="coverage",
            stretch=True, link=None, step_hint=None, warn=None) -> np.ndarray:
    # 기본은 coverage 다. 판정이 쓰는 zone 배열은 mask 를 거리로 되돌린 것이라 값이
    # 두 개뿐이고, 팔레트를 씌워도 2색 실루엣밖에 안 나온다.
    img = (layers or {}).get(layer)
    if layer == "zone" or img is None:
        zones, shown = render_zones(depth_mm, palette, cell, grid), "zone"
    elif layer == "coverage":
        zones, shown = render_coverage(img, palette, cell, grid, stretch), "coverage"
    else:
        zones, shown = render_plain(img, cell, grid), layer
    # 어느 레이어를 보고 있는지 항상 적어 둔다. 없어서 화면을 오해하기 쉬웠다.
    _text(zones, shown, (8, 20), 0.45)
    if warn:
        # 배경이 깨져도 판정은 태연히 라벨을 내놓는다. 그 조용한 오답이 제일 나쁘다.
        cv2.rectangle(zones, (0, 28), (zones.shape[1], 56), (0, 0, 120), -1)
        _text(zones, warn, (8, 48), 0.5, (120, 200, 255), 2)
    height = max(zones.shape[0], PANEL_MIN_H)
    if zones.shape[0] < height:            # zone 맵이 짧으면 위아래로 여백을 준다
        pad = height - zones.shape[0]
        top = pad // 2
        zones = cv2.copyMakeBorder(zones, top, pad - top, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(18, 18, 20))
    panel = render_panel(verdict, height, step=step, bg_ready=bg_ready,
                         base_ready=base_ready, fps=fps, link=link, step_hint=step_hint)
    return np.hstack([zones, panel])


def list_ports() -> None:
    """붙어 있는 포트를 찍고 어느 쪽이 Vision Stream 인지 짚어 준다.

    보드는 CDC 두 개짜리 복합 장치라 두 포트가 같은 설명을 달고 나온다. 인터페이스
    번호만이 둘을 가르고, Windows 는 그걸 MI_nn 또는 LOCATION 끝자리로 적는다.
    """
    import re
    import serial.tools.list_ports as lp

    print("사용 가능한 포트:")
    for p in lp.comports():
        hwid = (p.hwid or "").upper()
        role = ""
        if "1209:0001" in hwid or "1209&PID_0001" in hwid:
            m = re.search(r"MI_(\d+)", hwid) or re.search(r"LOCATION=\S*[.](\d+)", hwid)
            itf = int(m.group(1)) if m else None
            if itf == 0:
                role = "   <- 브리지(esptool). 이거 말고"
            elif itf == 2:
                role = "   <- Vision Stream. 이걸 --port 에 주세요"
        print(f"  {p.device:10s} {p.description}{role}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("camera", "esp"), default="camera",
                    help="zone 배열을 어디서 받을지. 판정·화면은 어느 쪽이든 같다")
    ap.add_argument("--camera", type=int, default=0, help="--source camera")
    ap.add_argument("--port", help="--source esp: 시리얼 포트 (예: COM8)")
    ap.add_argument("--baud", type=int, default=921600, help="--source esp: LINK_BAUD 와 맞출 것")
    ap.add_argument("--cell", type=int, default=14, help="zone 하나를 화면 몇 픽셀로 그릴지")
    ap.add_argument("--bg-frames", type=int, default=30)
    ap.add_argument("--baseline-frames", type=int, default=60)
    ap.add_argument("--outdir", default="snapshots")
    ap.add_argument("--log", default="posture_log.csv",
                    help="1/2/3 으로 자세를 표시하는 동안 특징을 여기 기록한다")
    args = ap.parse_args()

    # 센서 의존은 여기서만 들어온다. 둘 다 read() -> ZoneFrame 이라 아래 루프는
    # 어느 쪽인지 알 필요가 없다.
    if args.source == "esp":
        if not args.port:
            list_ports()      # 어느 포트인지가 첫 관문이라, 틀렸다고만 하지 않는다
            ap.error("--source esp 는 --port 가 필요하다 (예: --port COM5)")
        from esp_source import EspZoneSource
        stub = EspZoneSource(args.port, args.baud)
    else:
        from camera_source import CameraZoneSource
        stub = CameraZoneSource(camera=args.camera)
    tracker = PostureTracker()
    step = STEP_BACKGROUND
    print("STEP 1 - 창에서 SPACE 를 누르고 화면 밖으로 나가세요.")

    outdir = Path(args.outdir)
    # 기본은 웹캠 경로가 내던 그림 그대로다 - zone 거리 배열을 RAINBOW-HC 로.
    # 센서를 갈아끼워도 같은 그림이 나와야 눈이 헷갈리지 않는다. coverage 는 v 로,
    # 다른 팔레트는 c 로 언제든 볼 수 있다.
    palette, grid = 0, False
    layer = LAYERS.index("zone")      # 판정이 실제로 쓰는 배열이 기본
    stretch = True
    # 보드에 보낼 값의 거울. 보드가 실제로 들고 있는 값은 '/' 로 확인한다.
    threshold, opening, fill, invert = 40, 1, True, False
    gain, exposure, auto_exp, preview = 16, 300, False, False
    guard, bg_adapt = 24, True      # 스케치 기본값 (bg_guard, bg_period != 0)
    bg_phase, bg_at, bg_marks = None, 0.0, 0   # None / "clear" / "settle"
    fps, last = 0.0, time.perf_counter()
    tag, log_rows = None, []
    try:
        while True:
            zone = stub.read()
            if zone is None:
                print(f"프레임 취득 실패, 종료합니다. {stub.error or ''}")
                break
            now = time.perf_counter()
            verdict = tracker.update(zone, now)

            # 1단계 전반부: 센서 자신의 배경을 먼저 새로 잡는다. ESP 는 'b' 를 받으면
            # AEC 를 다시 돌려 노출을 잡고 배경을 뜨는데, 그게 끝나기 전에 zone 기준을
            # 뜨면 사람이 섞인 배경 위에 기준이 굳어 이후 판정이 전부 어긋난다.
            step_hint = None
            if bg_phase == "clear":
                left = max(bg_at - now, 0.0)
                step_hint = f"Step out of shot - {left:.0f}s"
                if now >= bg_at:
                    bg_marks = sum(BG_DONE_MARK in t for t in stub.log_lines())
                    stub.send("b")
                    bg_phase, bg_at = "settle", now
                    print("센서 배경 재캡처 중...")
            elif bg_phase == "settle":
                waited = now - bg_at
                step_hint = "Sensor re-exposing, capturing background..."
                # 완료 문구를 기다리되, 안 오는 센서(웹캠 스텁)도 있으므로 상한을 둔다.
                done = sum(BG_DONE_MARK in t for t in stub.log_lines()) > bg_marks
                if waited >= BG_SETTLE_MIN_S and (done or waited >= BG_SETTLE_MAX_S):
                    bg_phase = None
                    tracker.start_background(args.bg_frames)
                    print("zone 기준 거리 수집 중...")

            # 수집이 끝나면 다음 단계로 넘어간다.
            if step == STEP_BACKGROUND and tracker.background.captured:
                step = STEP_BASELINE
                print("STEP 2 - 바른 자세로 앉아서 SPACE 를 누르세요.")
            elif step == STEP_BASELINE and tracker.baseline is not None:
                step = STEP_LIVE
                print("판정 시작. 엎드리거나 뒤로 젖혀 보세요.")

            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            # 판정 구간은 무조건 기록한다. 키를 눌러야만 남기게 했더니 실제 실행에서
            # 매번 빠졌다. 1~4 로 붙이는 정답 라벨(tag)은 선택 사항으로 둔다.
            if step == STEP_LIVE:
                fe = verdict.features
                log_rows.append({
                    "tag": tag or "", "t": f"{now:.3f}", "label": verdict.label,
                    "occupancy": f"{fe.occupancy:.4f}", "top_row": f"{fe.top_row:.4f}",
                    "spread": f"{fe.spread:.4f}", "head_mm": f"{fe.head_mm:.1f}",
                    "dist_mm": f"{verdict.parts.get('dist_mm', 0.0):.1f}",
                    "slump": f"{verdict.parts.get('slump', 0.0):.3f}",
                    "recline": f"{verdict.parts.get('recline', 0.0):.3f}",
                    "drowsy": f"{verdict.parts.get('drowsy', 0.0):.3f}",
                    "nod_rate": f"{verdict.nod_rate:.2f}",
                    "nod_dip": f"{verdict.parts.get('nod_dip', 0.0):.4f}",
                    "nod_amp": f"{verdict.parts.get('nod_amp', 0.0):.4f}",
                    "phi": f"{verdict.phi:.3f}", "delta": f"{verdict.delta:.3f}",
                })

            pics = {"coverage": stub.coverage, "mask": stub.mask,
                    "skeleton": stub.skeleton}

            # 사람이라면 zone 의 절반을 넘게 채우면서 동시에 정지해 있을 수 없다.
            # 둘 다면 배경이 틀린 것이고, 그건 n 으로만 고쳐진다.
            fe = verdict.features
            warn = None
            if fe.occupancy >= BG_SUSPECT_OCC and fe.motion < BG_SUSPECT_MOTION:
                warn = f"background stale? {fe.occupancy*100:.0f}% filled, still - press n"

            # 링크 상태. 멈춰도 마지막 프레임이 계속 그려지므로 화면만 보고는 모른다.
            if stub.error:
                link = (False, f"link down: {stub.error}")
            elif getattr(stub, "stale", False):
                link = (False, "link stale - no mask from the board")
            else:
                lines = stub.log_lines()
                link = (True, lines[-1] if lines else "link ok")

            canvas = compose(zone.depth_mm, verdict, step=step, palette=palette,
                             cell=args.cell, grid=grid,
                             bg_ready=tracker.background.captured,
                             base_ready=tracker.baseline is not None, fps=fps,
                             layers=pics, layer=LAYERS[layer],
                             stretch=stretch, link=link, step_hint=step_hint, warn=warn)
            if tag:
                _text(canvas, f"REC {tag}  ({len(log_rows)})", (16, 28), 0.6, (70, 70, 245), 2)
            cv2.imshow("posture from zone map", canvas)

            # 렌즈가 가려졌는지 노출이 날아갔는지는 이 창에서만 보인다. 한 장에
            # ~208ms 를 먹으므로 켜 둔 동안 판정 스트림이 느려진다.
            if preview and stub.preview is not None:
                shot = cv2.resize(stub.preview, None, fx=4, fy=4,
                                  interpolation=cv2.INTER_NEAREST)
                cv2.imshow("camera", cv2.cvtColor(shot, cv2.COLOR_GRAY2BGR))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                if step == STEP_BACKGROUND:
                    if bg_phase is None:
                        bg_phase, bg_at = "clear", now + BG_CLEAR_S
                        print(f"{BG_CLEAR_S:.0f}초 안에 화면 밖으로 나가세요 "
                              "(SPACE 를 다시 누르면 즉시 캡처).")
                    elif bg_phase == "clear":
                        bg_at = now          # 이미 비어 있으면 기다릴 이유가 없다
                elif step == STEP_BASELINE:
                    tracker.start_baseline(args.baseline_frames)
                    print("자세 baseline 수집 중... 바른 자세를 유지하세요.")
            elif key == ord("n"):
                step = STEP_BACKGROUND
                tracker.background.captured = False
                bg_phase = None
                print("STEP 1 다시 - SPACE 를 누르고 화면 밖으로 나가세요.")
            elif key == ord("b"):
                step = STEP_BASELINE
                tracker.baseline = None
                print("STEP 2 다시 - 바른 자세로 앉아서 SPACE.")
            elif key in TAG_KEYS:
                tag = TAG_KEYS[key]
                print(f"기록 시작: {tag} (0 누르면 정지)")
            elif key == ord("0"):
                tag = None
                print("기록 정지")
            elif key == ord("c"):
                palette = (palette + 1) % len(PALETTES)
            elif key == ord("g"):
                grid = not grid
            elif key == ord("a"):
                stretch = not stretch
            elif key == ord("v"):
                # 센서가 안 보내주는 레이어는 건너뛴다. zone 은 항상 있다.
                for _ in range(len(LAYERS)):
                    layer = (layer + 1) % len(LAYERS)
                    if LAYERS[layer] == "zone" or pics.get(LAYERS[layer]) is not None:
                        break
                print(f"레이어: {LAYERS[layer]}")
            elif key == ord("s"):
                outdir.mkdir(parents=True, exist_ok=True)
                path = outdir / f"posture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), canvas)
                print(f"saved {path}")

            # --- 보드 튜닝. 값은 절대값으로 보낸다 (viewer.py 와 같은 문자열) ---
            elif key == ord("["):
                threshold = max(0, threshold - 5);    stub.send(f"t{threshold}")
                print(f"threshold {threshold or 'auto(otsu)'}")
            elif key == ord("]"):
                threshold = min(255, threshold + 5);  stub.send(f"t{threshold}")
                print(f"threshold {threshold}")
            elif key == ord("o"):
                opening = max(0, opening - 1);        stub.send(f"o{opening}")
            elif key == ord("O"):
                opening = min(4, opening + 1);        stub.send(f"o{opening}")
            elif key == ord("h"):
                fill = not fill;      stub.send(f"h{1 if fill else 0}")
            elif key == ord("i"):
                invert = not invert;  stub.send(f"i{1 if invert else 0}")
            elif key == ord("-"):
                gain = max(4, gain - 8);     stub.send(f"g{gain}")
                print(f"gain {gain}/16")
            elif key == ord("="):
                gain = min(248, gain + 8);   stub.send(f"g{gain}")
                print(f"gain {gain}/16")
            elif key in (ord(","), ord(".")):
                # 노출을 건드리면 보드가 배경을 새로 잡는다(화면 전체가 움직이므로).
                # 그러면 지금 기준도 같이 무효다 - 1단계를 다시 해야 한다.
                exposure = (max(0, exposure - 50) if key == ord(",")
                            else min(1200, exposure + 50))
                stub.send(f"e{exposure}")
                print(f"exposure {exposure} - 보드 배경이 리셋됩니다. n 으로 1단계부터.")
            elif key == ord("f"):
                # 배경 모델을 얼려 둔다. 조명이 고정된 자리에서는 흘러갈 이유가 없고,
                # 흘러가는 것이 가만히 있는 사람을 녹이는 원인이다.
                bg_adapt = not bg_adapt; stub.send(f"a{4 if bg_adapt else 0}")
                print(f"배경 적응 {'on' if bg_adapt else 'FROZEN'}")
            elif key in (ord("w"), ord("W")):
                # 흡수 가드. 이 폭을 넘게 어긋난 칸은 배경에 절대 흡수되지 않는다 —
                # 가만히 있는 사람을 지키는 장치이지만, 동시에 한 번 틀어진 배경이
                # 스스로 회복하지 못하는 이유이기도 하다. 그게 잔상이다.
                guard = max(0, guard - 4) if key == ord("w") else min(255, guard + 4)
                stub.send(f"s{guard}")
                print(f"흡수 가드 {guard}" + ("  (0=전부 흡수: 잔상은 사라지지만 "
                      "가만히 있는 사람도 녹는다)" if guard == 0 else ""))
            elif key == ord("x"):
                auto_exp = not auto_exp; stub.send(f"x{1 if auto_exp else 0}")
                print(f"auto exposure {'on' if auto_exp else 'off'} - n 으로 1단계부터.")
            elif key == ord("/"):
                stub.send("?")
            elif key == ord("p"):
                preview = not preview
                stub.send(f"p{4 if preview else 0}")   # 4프레임당 1장
                if not preview:
                    try:
                        cv2.destroyWindow("camera")
                    except cv2.error:
                        pass
    finally:
        stub.release()
        cv2.destroyAllWindows()
        if log_rows:
            import csv
            with open(args.log, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(log_rows[0]))
                w.writeheader()
                w.writerows(log_rows)
            print(f"{len(log_rows)} 행 기록: {args.log}")


if __name__ == "__main__":
    main()
