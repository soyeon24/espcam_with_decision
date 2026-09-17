"""54x42 zone depth map -> 자세 판정 -> posture Signal(phi/delta).

**센서 비의존 계층.** 입력은 VL53L9CX 가 내는 것과 같은 모양의 거리 배열 하나뿐이다.
카메라 스텁(tof_stub.py)을 실센서로 갈아끼워도 이 파일은 그대로 쓴다.
RGB·텍스처·얼굴 같은 카메라 고유 정보는 여기 들어오지 않는다.

판정하는 자세 (UPRIGHT 기준 대비):
  ABSENT   자리비움    - 점유 zone 이 거의 없다. FSM 의 present 로 나간다
  SLUMP    엎드림      - 이탈하면서 센서에 가까워진다
  RECLINE  뒤로 젖힘   - 이탈하면서 센서에서 멀어진다
  DROWSY   졸음        - 머리가 내려갔다 올라오기를 반복한다(꾸벅꾸벅)

엎드림과 졸음은 둘 다 머리가 내려가지만 **시간 패턴**이 다르다 - 엎드림은 내려가서
머무는 것, 졸음은 오르내림이 반복되는 것. 그래서 꾸벅임 횟수로 가른다. 꾸벅임 진폭은
사람마다·거리마다 달라서 본인 baseline 크기에 대한 비율로 잡는다.

**엎드림과 젖힘은 머리 높이로 구분할 수 없다.** 2D 투영에서는 둘 다 머리가 화면
아래로 내려간다(실측: 바른자세 head_row 0.098 -> 엎드림 0.341, 젖힘도 내려감).
그래서 머리 높이·몸 접힘은 '얼마나 이탈했나'(크기)만 정하고, '어느 쪽 이탈인가'는
거리 변화의 **부호**로 가른다 - 엎드리면 가까워지고(428->323mm) 젖히면 멀어진다.

사람 zone 은 절대 거리가 아니라 **배경 대비 침입량**으로 고른다. 실센서는 책상 상판과
벽을 항상 같이 보기 때문에, 절대 임계(예: 1.2m 이내)로 자르면 책상이 늘 사람으로 잡힌다.

주의: 임계값은 실 ToF 로그로 재튜닝해야 한다. 스텁의 거리값은 기하만 맞고
노이즈·책상 상판 반환 특성이 없다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# docs/agent-briefing.md C1 - ToF 는 VL53L9CX (54x42 zone).
ZONE_COLS, ZONE_ROWS = 54, 42

MAX_RANGE_MM = 4000.0    # 이보다 먼 값은 무효로 본다
INTRUSION_MM = 150.0     # 배경보다 이만큼 가까우면 사람 zone
BG_DECAY_MM = 1.0        # 자동 학습 시 배경이 프레임당 내려오는 폭
PRESENT_MIN_OCC = 0.04   # 재실로 볼 최소 점유율

SLUMP_SPAN = 0.22        # 머리가 baseline 대비 이만큼(세로 비율) 내려가면 이탈 최대
# 엎드림은 '내려가서 머무는' 것이다. 이 시간을 채워야 엎드림으로 인정한다.
# 이게 없으면 꾸벅임의 하강 순간마다 slump 가 치솟아 DROWSY 와 뒤섞인다
# (slump 는 순간값, drowsy 는 30초 창값이라 시간 축이 애초에 안 맞는다).
SLUMP_HOLD_S = 3.0
# 지속 타이머를 켜는 문턱. 라벨 문턱(0.70)에서 켜면 그보다 낮은 이탈이 전부 0 으로
# 죽어 delta 의 등급성이 사라진다(살짝 숙임 0.45 -> 0.00 이 됐었다).
SLUMP_ARM_AT = 0.25
# 젖힘도 '젖혀서 머무는' 것이다. 꾸벅임은 상체 전체가 잠깐 흔들려 거리 축에 스파이크를
# 남기는데, 라벨 우선순위에서 젖힘이 맨 앞이라 그 순간 DROWSY 가 통째로 가려졌다.
# 엎드림에 SLUMP_HOLD_S 를 둔 것과 같은 이유이고, 그때 젖힘에는 빠져 있었다.
# 값은 NOD_MAX_DOWN_S 에 맞춘다 - 꾸벅임으로 세어 주는 가장 긴 하강보다 오래
# 유지돼야 비로소 자세로 본다.
RECLINE_HOLD_S = 2.5
RECLINE_ARM_AT = 0.25    # 지속 타이머를 켜는 문턱. 라벨 문턱에서 켜면 등급성이 죽는다
RECLINE_SPAN_MM = 150.0  # 이만큼 멀어지면 젖힘 최대 (실측: 젖힘 +173mm, 엎드림 -105mm)
DIST_DEADBAND_MM = 40.0  # 거리 노이즈. 이 안이면 방향을 정하지 않는다
# 거리는 엎드림과 젖힘을 가르는 **유일한** 축이라 노이즈에 가장 약하다. 실센서가
# 아닌 경로에서는 겉보기 크기에서 역산하는데, 그 크기가 마스크 면적을 타고 프레임마다
# 출렁인다. 순간값으로 판정하면 두 자세가 섞이므로 이 창의 중앙값을 쓴다.
# 평균이 아니라 중앙값인 이유는 마스크가 한두 프레임씩 크게 튀기 때문이다.
DIST_SMOOTH_S = 1.2
BASELINE_CLIP_AT = 0.02  # baseline top_row 가 이보다 작으면 머리가 화각에 잘린 것
NOD_WINDOW_S = 30.0      # 꾸벅임 집계 창. FSM tick(30s)과 맞춰 둔다
# 꾸벅임 진폭은 절대값이 아니라 **본인 baseline 세로 크기에 대한 비율**이다.
# 사람마다 목 움직임이 다르고, 센서에서 멀어지면 같은 동작도 작게 찍히기 때문.
# 실측 근거: 42행이 화각을 덮으므로 한 행이 대략 3cm 다. 졸음 꾸벅임은 머리가
# 5~10cm 내려가니 2~3행이고, 0.10 x spread(0.7) = 3행이면 문턱이 꾸벅임 자체와
# 같은 크기다 - 대부분이 안 세어져 빈도가 쌓이지 않았다. 1~2행으로 내린다.
NOD_AMPLITUDE_REL = 0.06
NOD_AMPLITUDE_MIN = 0.03  # baseline 이 없거나 지나치게 작을 때의 하한
NOD_MIN_DOWN_S = 0.30    # 내려가 있던 시간이 이보다 짧으면 노이즈로 버린다
# 조는 꾸벅임은 잠깐 떨어졌다 홱 올라온다. 키보드를 내려다보는 건 몇 초씩 머문다.
# 그래서 오래 내려가 있던 하강은 꾸벅임으로 세지 않는다.
NOD_MAX_DOWN_S = 2.5
NOD_REFRACTORY_S = 0.8   # 직전 꾸벅임 이후 최소 간격
DIP_LOOKBACK_S = 3.0     # 화면에 보여줄 '가장 깊었던 하강'을 찾는 창
NOD_RATE_FULL = 12.0     # 분당 이 횟수면 졸음 기여도 최대 (라벨은 그 절반인 6회/분)
# 관측 시간이 짧을 때 그 시간으로 나누면 빈도가 커진다 - 꾸벅임 1회를 2초로
# 나누면 30회/분. 분모에 하한을 둬서 단발 숙임이 졸음으로 읽히지 않게 한다.
NOD_MIN_SPAN_S = 12.0
# 졸음은 '지금' 상태다. 꾸벅임을 멈췄는데 이벤트가 30초 창에서 빠지길 기다리면
# 깨서 가만히 있어도 DROWSY 가 최대 30초 남는다. 그래서 마지막 꾸벅임 이후
# 조용하면 창과 무관하게 기여도를 내린다.
NOD_QUIET_S = 8.0        # 이만큼 조용하면 감쇠 시작
NOD_FADE_S = 6.0         # 이 시간에 걸쳐 0 으로 (총 14초면 완전히 깨어난 것으로 본다)
MOTION_SPAN = 0.06       # 움직임 정규화 기준

# 머리 찾기 - '맨 위 점유 행'을 머리로 쓰면 손을 든 순간 그 손이 머리가 된다.
# 머리는 폭이 넓고(팔뚝은 1~2 zone) 몸통 중심축 위에 있다는 두 조건으로 가른다.
HEAD_MIN_WIDTH = 4       # 머리로 인정할 최소 가로 zone 수
HEAD_BAND_ROWS = 4       # 머리 거리 산출에 쓸 행 수
TORSO_HALF_SPAN = 0.22   # 몸통 중심에서 이 비율(가로) 안쪽만 머리 후보로 본다

SLUMP_LABEL_AT = 0.70    # 라벨을 SLUMP 로 붙일 기여도
RECLINE_LABEL_AT = 0.55
DROWSY_LABEL_AT = 0.50


def _finite(depth_mm: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(depth_mm), depth_mm, MAX_RANGE_MM)


@dataclass
class ZoneFrame:
    """한 프레임의 zone 거리 배열 (mm). NaN 은 무효 zone."""
    depth_mm: np.ndarray


@dataclass
class BackgroundModel:
    """zone 별 정적 배경 거리(책상 상판·벽).

    책상을 비우고 capture() 하는 것이 정확하다. 못 했을 때를 위해 zone 별
    '천천히 감쇠하는 최댓값'으로 자동 학습한다 - 사람은 zone 을 가깝게만 만든다는
    성질을 쓰지만, 오래 완전히 정지해 있으면 사람이 배경으로 흡수된다.
    """
    ref_mm: np.ndarray | None = None
    captured: bool = False

    def capture(self, frames: list[np.ndarray]) -> None:
        if not frames:
            raise ValueError("배경을 만들 프레임이 없다")
        self.ref_mm = np.median(np.stack([_finite(f) for f in frames]), axis=0)
        self.captured = True

    def update(self, depth_mm: np.ndarray) -> None:
        if self.captured:
            return
        d = _finite(depth_mm)
        if self.ref_mm is None or self.ref_mm.shape != d.shape:
            self.ref_mm = np.full(d.shape, MAX_RANGE_MM, np.float64)
        self.ref_mm = np.maximum(self.ref_mm - BG_DECAY_MM, d)

    def occupied(self, depth_mm: np.ndarray) -> np.ndarray:
        if self.ref_mm is None:
            return np.zeros(depth_mm.shape, bool)
        return (self.ref_mm - _finite(depth_mm)) >= INTRUSION_MM


@dataclass
class PostureFeatures:
    occupancy: float      # 점유 zone 비율 [0,1]
    top_row: float        # 최상단 점유 행 (0=위, 1=아래) - 머리 높이
    centroid_row: float   # 점유 무게중심 행
    spread: float         # 세로 점유 범위
    head_mm: float        # 머리 zone 거리 중앙값 (mm)
    motion: float         # 직전 프레임 대비 변화 zone 비율
    # 머리 밴드에서 가장 넓은 행 / 화면 폭. 겉보기 크기를 재는 축인데, 면적과 달리
    # 의자에 오염되지 않는다 - 등받이는 머리 아래에 있고, 사람과 같이 움직이며
    # 면적을 메꿔서 젖힘과 엎드림을 구별할 수 없게 만든다.
    head_w: float = 0.0


@dataclass
class PostureVerdict:
    label: str            # ABSENT/UPRIGHT/SLUMP/RECLINE/DROWSY/BASELINE/UNKNOWN
    present: bool         # FSM SensorFrame.present 로 나간다
    phi: float            # 집중 기여도 [0,1]
    delta: float          # 피로 기여도 [0,1]
    features: PostureFeatures
    parts: dict[str, float] = field(default_factory=dict)   # 자세별 기여도
    nod_rate: float = 0.0                                   # 분당 꾸벅임
    note: str = ""


def find_head(occ: np.ndarray) -> tuple[int, int, int] | None:
    """머리 행과 몸통 중심 열 구간을 찾는다. 반환: (행, 열_시작, 열_끝) 또는 None.

    손을 들면 그 손이 최상단 점유 행이 되어 '머리가 올라갔다'로 읽히고, 머리 거리도
    손 거리로 바뀐다. 그래서 최상단이 아니라 **폭이 충분하고 몸통 중심축 위에 있는**
    가장 높은 행을 머리로 본다.
    """
    counts = occ.sum(axis=1)
    if not counts.any():
        return None
    rows, cols = occ.shape

    # 몸통 = 가장 넓은 행들. 그 열 무게중심을 중심축으로 쓴다.
    body = np.flatnonzero(counts >= max(counts.max() * 0.5, 1.0))
    col_w = occ[body].sum(axis=0).astype(np.float64)
    if col_w.sum() <= 0:
        return None
    center = float(col_w @ np.arange(cols) / col_w.sum())
    half = max(TORSO_HALF_SPAN * cols, HEAD_MIN_WIDTH)
    lo, hi = int(max(center - half, 0)), int(min(center + half, cols))

    band = occ[:, lo:hi].sum(axis=1)
    cand = np.flatnonzero(band >= HEAD_MIN_WIDTH)
    if cand.size == 0:
        return None
    return int(cand[0]), lo, hi


def extract(frame: ZoneFrame, background: BackgroundModel,
            prev: np.ndarray | None = None) -> tuple[PostureFeatures, np.ndarray]:
    """zone 배열에서 기하 특징을 뽑는다. 반환: (특징, 이번 점유 마스크)."""
    background.update(frame.depth_mm)
    occ = background.occupied(frame.depth_mm)
    rows, _ = occ.shape
    total = float(occ.size)
    occupancy = float(occ.sum()) / total
    depth = _finite(frame.depth_mm)

    head = find_head(occ)
    row_any = occ.any(axis=1)
    if head is not None and row_any.any():
        top, lo, hi = head
        bottom = int(np.flatnonzero(row_any)[-1])
        top_row = top / (rows - 1)
        spread = max(bottom - top, 0) / (rows - 1)
        weights = occ.sum(axis=1).astype(np.float64)
        centroid_row = float((weights @ np.arange(rows)) / weights.sum()) / (rows - 1)
        # 머리 거리 = 머리 행부터 몇 줄, 몸통 중심 열 구간 안쪽만. 들어올린 손이
        # 섞이지 않게 열도 같이 제한한다. 뒤로 젖힘을 잡는 축이라 오염되면 치명적이다.
        end = min(top + HEAD_BAND_ROWS, rows)
        mask = occ[top:end, lo:hi]
        vals = depth[top:end, lo:hi][mask]
        # 평균이 아니라 중앙값 - 머리 바로 위로 손을 들면 팔뚝이 소수 픽셀로 섞이는데,
        # 평균은 그걸 따라가고(실측 -69mm) 중앙값은 버린다.
        head_mm = float(np.median(vals)) if vals.size else float("nan")
        head_w = float(mask.sum(axis=1).max()) / occ.shape[1] if mask.size else 0.0
    else:
        top_row = centroid_row = spread = 1.0
        head_mm = float("nan")
        head_w = 0.0

    motion = 0.0 if prev is None or prev.shape != occ.shape else \
        float(np.logical_xor(occ, prev).sum()) / total
    return (PostureFeatures(occupancy, top_row, centroid_row, spread, head_mm,
                            motion, head_w), occ)


@dataclass
class NodDetector:
    """머리 높이의 '내려갔다 올라오기'를 세어 분당 꾸벅임을 낸다.

    FSM tick 이 30초라 engine 은 개별 꾸벅임을 볼 수 없다. 그래서 집계는 반드시
    이 특징 계층에서 끝내고 delta 에 실어 보내야 한다.

    기준선은 창의 중앙값이라 자세가 통째로 바뀌어도 따라간다. 그래서 엎드려서
    머물면(중앙값이 같이 내려감) 꾸벅임으로 세지 않는다 - 엎드림과 졸음이 갈리는 지점.
    """
    window_s: float = NOD_WINDOW_S
    amplitude: float = NOD_AMPLITUDE_MIN   # update() 에서 baseline 크기에 맞춰 갱신
    _hist: deque = field(default_factory=deque, repr=False)
    _events: deque = field(default_factory=deque, repr=False)
    _down_since: float | None = None
    _last_event: float = -1e9
    _abandoned: bool = False
    # 진단용. 꾸벅임이 안 세어질 때 '문턱에 얼마나 모자랐나'를 볼 수 있어야 한다.
    rest: float = float("nan")      # 창의 중앙값 = 지금의 기준 머리 높이
    dip: float = 0.0                # 최근 DIP_LOOKBACK_S 안에서 가장 깊이 내려간 값

    def reset(self) -> None:
        self._hist.clear()
        self._events.clear()
        self._down_since = None
        self._abandoned = False
        self.rest, self.dip = float("nan"), 0.0

    def update(self, now: float, top_row: float, present: bool, scale: float = 1.0) -> float:
        # 사람이 없으면 top_row 가 1.0 과 잔여값 사이를 튀어 가짜 꾸벅임이 쌓인다
        # (실측: 자리비움 상태에서 nod/min 40).
        if not present:
            self.reset()
            return 0.0

        self.amplitude = max(NOD_AMPLITUDE_REL * scale, NOD_AMPLITUDE_MIN)
        self._hist.append((now, top_row))
        while self._hist and now - self._hist[0][0] > self.window_s:
            self._hist.popleft()
        while self._events and now - self._events[0] > self.window_s:
            self._events.popleft()

        if len(self._hist) >= 8:
            rest = float(np.median([v for _, v in self._hist]))
            self.rest = rest
            # 지금 값이 아니라 최근 창에서 가장 깊었던 하강을 보여준다. 고개를 든
            # 뒤에 읽으면 0 이라, 방금 꾸벅임이 문턱에 얼마나 모자랐는지 알 수 없다.
            recent = [v for ts, v in self._hist if now - ts <= DIP_LOOKBACK_S]
            self.dip = (max(recent) - rest) if recent else 0.0
            up = top_row < rest + self.amplitude * 0.3
            if self._abandoned:
                if up:
                    self._abandoned = False                     # 다시 올라와야 재무장
            elif self._down_since is None:
                if top_row > rest + self.amplitude:
                    self._down_since = now                      # 머리가 내려갔다
            elif up:
                held = now - self._down_since
                self._down_since = None                         # 다시 올라왔다
                if (NOD_MIN_DOWN_S <= held <= NOD_MAX_DOWN_S
                        and now - self._last_event >= NOD_REFRACTORY_S):
                    self._events.append(now)                    # 꾸벅 1회
                    self._last_event = now
            elif now - self._down_since > NOD_MAX_DOWN_S:
                # 계속 내려가 있다 = 응시/엎드림. 올라올 때 뒤늦게 세지 않도록 무효화.
                self._down_since = None
                self._abandoned = True

        # 창이 아직 안 찼으면 관측 시간으로 나눠 초반 과대평가를 막는다.
        span = max(now - self._hist[0][0], 1.0) if self._hist else self.window_s
        rate = len(self._events) * 60.0 / max(min(span, self.window_s), NOD_MIN_SPAN_S)

        # 조용해지면 창에서 빠지길 기다리지 않고 내린다 - 깨어난 것을 바로 반영한다.
        quiet = now - self._last_event
        fade = float(np.clip(1.0 - (quiet - NOD_QUIET_S) / NOD_FADE_S, 0.0, 1.0))
        return rate * fade


@dataclass
class PostureBaseline:
    """바른 자세 기준값. FSM 의 START 단계 baseline 에 대응한다."""
    top_row: float
    spread: float
    head_mm: float
    samples: int = 0
    clipped: bool = False   # 머리가 화각 위쪽에 잘린 채로 측정됐다

    @classmethod
    def from_features(cls, feats: list[PostureFeatures]) -> "PostureBaseline":
        if not feats:
            raise ValueError("baseline 을 만들 특징이 없다")
        med = lambda a: float(np.median([getattr(f, a) for f in feats]))
        heads = [f.head_mm for f in feats if np.isfinite(f.head_mm)]
        top = med("top_row")
        # top_row 가 0 에 붙으면 머리가 잘린 것이다. 그 baseline 을 쓰면 바른 자세도
        # '머리가 내려갔다'로 읽혀 계속 이탈로 나온다(실측: 바른자세 slump 0.37).
        return cls(top, med("spread"),
                   float(np.median(heads)) if heads else float("nan"),
                   len(feats), clipped=top < BASELINE_CLIP_AT)


def judge(feats: PostureFeatures, base: PostureBaseline | None, nod_rate: float,
          slump_held_s: float = SLUMP_HOLD_S,
          head_mm: float | None = None,
          recline_held_s: float = RECLINE_HOLD_S) -> PostureVerdict:
    """특징 + baseline + 꾸벅임 빈도 -> 자세 라벨과 phi/delta.

    slump_held_s 는 머리가 연속으로 내려가 있던 시간. 짧으면 꾸벅임의 하강 국면일
    뿐이라 엎드림 기여도를 비례해서 깎는다.

    head_mm 을 주면 거리 축에만 그 값을 쓴다 (평활한 값). feats 안의 값은 날것으로
    남겨 로그에 그대로 남는다 - 평활이 과했는지 나중에 되짚을 수 있어야 한다.
    """
    dist_mm = feats.head_mm if head_mm is None else head_mm
    if feats.occupancy < PRESENT_MIN_OCC:
        return PostureVerdict("ABSENT", False, 0.0, 0.0, feats,
                              nod_rate=nod_rate, note="low occupancy")
    if base is None:
        return PostureVerdict("UNKNOWN", True, 0.0, 0.0, feats,
                              nod_rate=nod_rate, note="no baseline")

    # 1) 이탈 '크기' - 머리가 내려간 정도 + 몸이 접힌 정도. 방향 정보는 없다.
    head_drop = feats.top_row - base.top_row
    drop = float(np.clip(head_drop / SLUMP_SPAN, 0.0, 1.0))
    collapse = float(np.clip((base.spread - feats.spread) / max(base.spread, 1e-6), 0.0, 1.0))
    magnitude = float(np.clip(0.8 * drop + 0.2 * collapse, 0.0, 1.0))

    # 2) 젖힘은 '멀어진 거리' 자체가 고유 축이다. 머리 높이에 곱하면 안 된다 -
    #    머리를 안 내리고 젖히는 경우가 0 으로 사라진다.
    if np.isfinite(dist_mm) and np.isfinite(base.head_mm):
        dist_delta = dist_mm - base.head_mm
        recline_raw = float(np.clip((dist_delta - DIST_DEADBAND_MM) / RECLINE_SPAN_MM, 0.0, 1.0))
    else:
        dist_delta, recline_raw = 0.0, 0.0
    # 지속 조건을 못 채운 젖힘은 꾸벅임의 흔들림일 뿐이다.
    recline = recline_raw * float(np.clip(recline_held_s / RECLINE_HOLD_S, 0.0, 1.0))

    # 3) 엎드림은 머리 높이로 재되, 멀어지고 있으면 눌러 끈다. 2D 에선 젖혀도 머리가
    #    내려가므로, 이 게이트가 없으면 젖힘이 전부 엎드림으로 빨려 들어간다.
    #    게이트에는 순간값을 쓴다 - '어느 쪽으로 가고 있나'는 지금 정보이고, 여기에
    #    지속 조건까지 걸면 젖히는 3초 동안 엎드림이 먼저 문턱을 넘는다.
    slump_raw = magnitude * (1.0 - recline_raw)
    # 지속 시간이 안 찼으면 아직 엎드림이 아니다 - 꾸벅임의 하강 국면과 구분되는 지점.
    slump = slump_raw * float(np.clip(slump_held_s / SLUMP_HOLD_S, 0.0, 1.0))

    # dist_mm 은 바 없이 숫자로만 보여준다 - 지금 판정의 근거를 눈으로 확인하는 값.
    # 4) 졸음 - 꾸벅임 빈도. 자세가 아니라 시간 패턴이라 별도 축이다.
    drowsy = float(np.clip(nod_rate / NOD_RATE_FULL, 0.0, 1.0))

    parts = {"slump": slump, "recline": recline, "drowsy": drowsy,
             "dist_mm": dist_delta, "slump_raw": slump_raw, "recline_raw": recline_raw}
    # 뒤로 젖힘은 자세 불량이지만 각성 상태일 수 있어 피로 기여를 낮게 잡는다.
    delta = float(np.clip(max(0.95 * slump, 0.85 * drowsy, 0.45 * recline), 0.0, 1.0))
    stability = 1.0 - float(np.clip(feats.motion / MOTION_SPAN, 0.0, 1.0))
    phi = float(np.clip((1.0 - delta) * stability, 0.0, 1.0))

    # 젖힘을 먼저 본다. 반대로 두면 slump 가 먼저 문턱을 넘어 젖힘이 영영 안 뜬다
    # (실측에서 recline 1.0 인데도 라벨이 SLUMP 로 나왔던 버그).
    # 엎드림이 졸음보다 앞이다. slump 가 지속 조건(SLUMP_HOLD_S)을 통과해야만 올라오므로
    # 꾸벅임의 하강 국면은 여기까지 못 온다 - 순간값과 창값이 섞이던 문제가 여기서 끊긴다.
    if recline >= RECLINE_LABEL_AT:
        label = "RECLINE"
    elif slump >= SLUMP_LABEL_AT:
        label = "SLUMP"
    elif drowsy >= DROWSY_LABEL_AT:
        label = "DROWSY"
    else:
        label = "UPRIGHT"
    note = "baseline clipped - press b" if base.clipped else ""
    return PostureVerdict(label, True, phi, delta, feats, parts, nod_rate, note)


@dataclass
class PostureTracker:
    """프레임을 계속 넣으면 배경·baseline 수집과 판정을 이어서 해준다."""
    background: BackgroundModel = field(default_factory=BackgroundModel)
    baseline: PostureBaseline | None = None
    nods: NodDetector = field(default_factory=NodDetector)
    _prev: np.ndarray | None = field(default=None, repr=False)
    _posture_buf: list[PostureFeatures] = field(default_factory=list, repr=False)
    _bg_buf: list[np.ndarray] = field(default_factory=list, repr=False)
    _head_hist: deque = field(default_factory=deque, repr=False)
    _want_posture: int = 0
    _want_bg: int = 0
    _slump_since: float | None = field(default=None, repr=False)
    _recline_since: float | None = field(default=None, repr=False)

    def start_background(self, samples: int = 30) -> None:
        """책상을 비운 상태에서 호출할 것."""
        self._bg_buf, self._want_bg = [], samples

    def start_baseline(self, samples: int = 60) -> None:
        """바른 자세로 앉은 상태에서 호출할 것."""
        self._posture_buf, self._want_posture = [], samples

    @property
    def ready(self) -> bool:
        return self.baseline is not None and self._want_bg == 0 and self._want_posture == 0

    def update(self, frame: ZoneFrame, now: float) -> PostureVerdict:
        if self._want_bg > 0:
            self._bg_buf.append(frame.depth_mm)
            want, done = self._want_bg, len(self._bg_buf)
            if done >= want:
                self.background.capture(self._bg_buf)
                self._want_bg = 0
            feats, self._prev = extract(frame, self.background, self._prev)
            return PostureVerdict("BASELINE", False, 0.0, 0.0, feats,
                                  note=f"background {done}/{want}")

        feats, self._prev = extract(frame, self.background, self._prev)
        present = feats.occupancy >= PRESENT_MIN_OCC
        scale = self.baseline.spread if self.baseline else feats.spread
        nod_rate = self.nods.update(now, feats.top_row, present, scale)

        # 거리 축만 평활한다. 여기만 부호로 엎드림/젖힘을 가르므로 노이즈에 제일 약하고,
        # 자리를 비우면 남은 값이 의미를 잃으므로 같이 비운다.
        if not present:
            self._head_hist.clear()
        elif np.isfinite(feats.head_mm):
            self._head_hist.append((now, feats.head_mm))
        while self._head_hist and now - self._head_hist[0][0] > DIST_SMOOTH_S:
            self._head_hist.popleft()
        head_mm = (float(np.median([v for _, v in self._head_hist]))
                   if self._head_hist else float("nan"))

        if self._want_posture > 0:
            want = self._want_posture
            if feats.occupancy >= PRESENT_MIN_OCC:
                self._posture_buf.append(feats)
            done = len(self._posture_buf)
            if done >= want:
                self.baseline = PostureBaseline.from_features(self._posture_buf)
                self._want_posture = 0
            return PostureVerdict("BASELINE", True, 0.0, 0.0, feats,
                                  nod_rate=nod_rate, note=f"posture {done}/{want}")

        # 머리가 연속으로 내려가 있던 시간. 직전 프레임의 raw 값으로 재는 한 프레임
        # 지연이 있지만 30fps 에서는 무시할 수 있다.
        held = (now - self._slump_since) if self._slump_since is not None else 0.0
        held_r = (now - self._recline_since) if self._recline_since is not None else 0.0
        verdict = judge(feats, self.baseline, nod_rate, held, head_mm, held_r)

        if verdict.parts.get("slump_raw", 0.0) >= SLUMP_ARM_AT:
            if self._slump_since is None:
                self._slump_since = now
        else:
            self._slump_since = None
        # 꾸벅임이 왜 안 세어지는지 보이게 같이 실어 보낸다. nod/min 이 0 인 것만
        # 보여서는 진폭이 모자란 건지 애초에 안 움직인 건지 알 수가 없었다.
        verdict.parts["nod_dip"] = self.nods.dip
        verdict.parts["nod_amp"] = self.nods.amplitude

        if verdict.parts.get("recline_raw", 0.0) >= RECLINE_ARM_AT:
            if self._recline_since is None:
                self._recline_since = now
        else:
            self._recline_since = None
        return verdict
