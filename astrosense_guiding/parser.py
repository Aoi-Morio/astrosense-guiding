"""PHD2 guide-log parser.

PHD2 writes one text file per night (``PHD2_GuideLog_YYYY-MM-DD_HHMMSS.txt``).
A file contains any number of *Calibration* and *Guiding* sections.  Each
guiding section has a free-form header (key = value pairs), a CSV column line
beginning with ``Frame,`` and then one CSV row per guide frame interleaved with
``INFO:`` event lines (dither, settling, parameter changes ...).

Only guiding sections are returned as :class:`GuideSession` objects; the
number of calibration sections is kept for diagnostics.
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Column name aliases across PHD2 versions -> canonical name
_COLUMN_ALIASES = {
    "frame": "frame",
    "time": "time",
    "mount": "mount",
    "dx": "dx",
    "dy": "dy",
    "rarawdistance": "ra_raw",
    "radistanceraw": "ra_raw",
    "decrawdistance": "dec_raw",
    "decdistanceraw": "dec_raw",
    "raguidedistance": "ra_guide",
    "radistanceguide": "ra_guide",
    "decguidedistance": "dec_guide",
    "decdistanceguide": "dec_guide",
    "raduration": "ra_dur",
    "radirection": "ra_dir",
    "decduration": "dec_dur",
    "decdirection": "dec_dir",
    "xstep": "xstep",
    "ystep": "ystep",
    "starmass": "star_mass",
    "snr": "snr",
    "errorcode": "err",
}


@dataclass
class GuideEvent:
    """An ``INFO:`` line inside a guiding section."""

    frame_index: int  # index into the session arrays of the *preceding* frame (-1 = before first)
    time: float  # seconds since guiding start (time of preceding frame)
    kind: str  # "dither", "settle_start", "settle_done", "settle_fail", "param", "other"
    text: str
    dither_dx: float | None = None
    dither_dy: float | None = None


@dataclass
class GuideSession:
    """One ``Guiding Begins`` ... ``Guiding Ends`` block."""

    index: int
    start: str
    header_lines: list[str]
    params: dict
    frame: np.ndarray
    t: np.ndarray
    mount: np.ndarray  # str: "Mount", "AO", "DROP"
    ra_raw: np.ndarray  # px
    dec_raw: np.ndarray  # px
    ra_guide: np.ndarray  # px
    dec_guide: np.ndarray  # px
    ra_dur: np.ndarray  # ms
    ra_dir: np.ndarray  # str
    dec_dur: np.ndarray  # ms
    dec_dir: np.ndarray  # str
    star_mass: np.ndarray
    snr: np.ndarray
    err: np.ndarray
    events: list[GuideEvent] = field(default_factory=list)

    # ---- convenience -------------------------------------------------
    @property
    def n_frames(self) -> int:
        return int(self.t.size)

    @property
    def valid(self) -> np.ndarray:
        """Frames with a usable star measurement."""
        return (self.mount != "DROP") & np.isfinite(self.ra_raw) & np.isfinite(self.dec_raw)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size > 1 else 0.0

    @property
    def cadence(self) -> float:
        """Median time between frames [s]."""
        if self.t.size < 3:
            return float("nan")
        return float(np.median(np.diff(self.t)))

    @property
    def pixel_scale(self) -> float:
        return float(self.params.get("pixel_scale", 1.0))

    @property
    def dec_deg(self) -> float | None:
        return self.params.get("dec_deg")

    def summary(self) -> str:
        nv = int(self.valid.sum())
        return (f"#{self.index} {self.start}  {self.n_frames} frames ({nv} valid), "
                f"{self.duration/60:.1f} min, cadence {self.cadence:.2f} s")


@dataclass
class GuideLog:
    path: str
    version: str
    sessions: list[GuideSession]
    n_calibrations: int
    n_calibration_star_lost: int

    def longest(self) -> GuideSession | None:
        good = [s for s in self.sessions if s.n_frames > 0]
        return max(good, key=lambda s: int(s.valid.sum())) if good else None


# ---------------------------------------------------------------------------
_NUM = r"([-+]?\d+(?:\.\d+)?)"
_HEADER_PATTERNS = {
    "pixel_scale": re.compile(r"Pixel scale = " + _NUM + r" arc-sec/px"),
    "binning": re.compile(r"Binning = (\d+)"),
    "focal_length_mm": re.compile(r"Focal length = " + _NUM + r" mm"),
    "exposure_ms": re.compile(r"Exposure = (\d+) ms"),
    "x_angle_deg": re.compile(r"xAngle = " + _NUM),
    "x_rate_px_s": re.compile(r"xRate = " + _NUM),
    "y_angle_deg": re.compile(r"yAngle = " + _NUM),
    "y_rate_px_s": re.compile(r"yRate = " + _NUM),
    "ra_guide_speed": re.compile(r"RA Guide Speed = " + _NUM + r" a-s/s"),
    "dec_guide_speed": re.compile(r"Dec Guide Speed = " + _NUM + r" a-s/s"),
    "ra_hr": re.compile(r"(?:^|, )RA = " + _NUM + r" hr"),
    "dec_deg": re.compile(r"(?:^|, )Dec = " + _NUM + r" deg"),
    "hour_angle_hr": re.compile(r"Hour angle = " + _NUM + r" hr"),
    "alt_deg": re.compile(r"Alt = " + _NUM + r" deg"),
    "max_ra_ms": re.compile(r"Max RA duration = (\d+)"),
    "max_dec_ms": re.compile(r"Max DEC duration = (\d+)"),
    "backlash_pulse_ms": re.compile(r"Backlash comp = \w+, pulse = (\d+) ms"),
}
_HEADER_STR_PATTERNS = {
    "mount": re.compile(r"^Mount = ([^,]+)"),
    "profile": re.compile(r"Equipment Profile = (.+)$"),
    "pier_side": re.compile(r"Pier side = (\w+)"),
    "x_algorithm": re.compile(r"^X guide algorithm = (.+)$"),
    "y_algorithm": re.compile(r"^Y guide algorithm = (.+)$"),
    "dec_guide_mode": re.compile(r"DEC guide mode = (\w+)"),
    "backlash_comp": re.compile(r"Backlash comp = (\w+)"),
    "camera": re.compile(r"Camera = ([^,]+)"),
}
_DITHER_RE = re.compile(r"DITHER by " + _NUM + r",\s*" + _NUM)


def _parse_header(lines: list[str]) -> dict:
    params: dict = {}
    for line in lines:
        s = line.strip()
        for key, pat in _HEADER_PATTERNS.items():
            if key not in params:
                m = pat.search(s)
                if m:
                    params[key] = float(m.group(1))
        for key, pat in _HEADER_STR_PATTERNS.items():
            if key not in params:
                m = pat.search(s)
                if m:
                    params[key] = m.group(1).strip()
    return params


def _to_float(v: str | None) -> float:
    if v is None:
        return math.nan
    v = v.strip().strip('"')
    if not v:
        return math.nan
    try:
        return float(v)
    except ValueError:
        return math.nan


def _classify_info(text: str) -> tuple[str, float | None, float | None]:
    up = text.upper()
    if "DITHER" in up:
        m = _DITHER_RE.search(text)
        if m:
            return "dither", float(m.group(1)), float(m.group(2))
        return "dither", None, None
    if "SETTLING STATE CHANGE" in up:
        if "STARTED" in up:
            return "settle_start", None, None
        if "COMPLETE" in up:
            return "settle_done", None, None
        if "FAILED" in up:
            return "settle_fail", None, None
    if "PARAMETER CHANGE" in up:
        return "param", None, None
    return "other", None, None


class _SessionBuilder:
    def __init__(self, index: int, start: str):
        self.index = index
        self.start = start
        self.header: list[str] = []
        self.columns: list[str] | None = None
        self.rows: list[dict] = []
        self.events: list[GuideEvent] = []

    def add_row(self, line: str) -> None:
        assert self.columns is not None
        vals = next(csv.reader(io.StringIO(line)))
        row = {}
        for name, v in zip(self.columns, vals):
            row[name] = v
        self.rows.append(row)

    def add_info(self, text: str) -> None:
        kind, ddx, ddy = _classify_info(text)
        idx = len(self.rows) - 1
        t = _to_float(self.rows[-1].get("time")) if self.rows else 0.0
        self.events.append(GuideEvent(idx, t, kind, text, ddx, ddy))

    def build(self) -> GuideSession:
        n = len(self.rows)
        col = lambda k: [r.get(k) for r in self.rows]  # noqa: E731
        f = lambda k: np.array([_to_float(v) for v in col(k)], dtype=float)  # noqa: E731
        s = lambda k: np.array([(v or "").strip().strip('"') for v in col(k)], dtype=object)  # noqa: E731
        ra_dur = np.nan_to_num(f("ra_dur"), nan=0.0)
        dec_dur = np.nan_to_num(f("dec_dur"), nan=0.0)
        return GuideSession(
            index=self.index, start=self.start, header_lines=self.header,
            params=_parse_header(self.header),
            frame=f("frame").astype(float), t=f("time"),
            mount=s("mount") if n else np.array([], dtype=object),
            ra_raw=f("ra_raw"), dec_raw=f("dec_raw"),
            ra_guide=f("ra_guide"), dec_guide=f("dec_guide"),
            ra_dur=ra_dur, ra_dir=s("ra_dir"), dec_dur=dec_dur, dec_dir=s("dec_dir"),
            star_mass=f("star_mass"), snr=f("snr"), err=f("err"),
            events=self.events,
        )


def parse_text(text: str, path: str = "<memory>") -> GuideLog:
    version = ""
    sessions: list[GuideSession] = []
    n_cal = 0
    n_cal_lost = 0
    cur: _SessionBuilder | None = None
    in_cal = False

    def close():
        nonlocal cur
        if cur is not None:
            sessions.append(cur.build())
            cur = None

    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        st = line.strip()
        if not version and st.startswith("PHD2 version"):
            version = st
            continue
        if st.startswith("Guiding Begins at"):
            close()
            in_cal = False
            cur = _SessionBuilder(len(sessions), st[len("Guiding Begins at"):].strip())
            continue
        if st.startswith("Calibration Begins at"):
            close()
            in_cal = True
            n_cal += 1
            continue
        if st.startswith("Guiding Ends at") or st.startswith("Log closed") or st.startswith("Log Summary"):
            close()
            in_cal = False
            continue
        if in_cal:
            if "STAR LOST" in st.upper():
                n_cal_lost += 1
            continue
        if cur is None:
            continue
        if st.startswith("Frame,"):
            names = next(csv.reader(io.StringIO(st)))
            cur.columns = [_COLUMN_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in names]
            continue
        if cur.columns is None:
            if st:
                cur.header.append(st)
            continue
        if st.startswith("INFO:"):
            cur.add_info(st[5:].strip())
            continue
        if st and st[0].isdigit():
            cur.add_row(st)
    close()
    return GuideLog(path=path, version=version, sessions=sessions,
                    n_calibrations=n_cal, n_calibration_star_lost=n_cal_lost)


def parse_file(path: str | Path) -> GuideLog:
    p = Path(path)
    data = p.read_bytes()
    for enc in ("utf-8-sig", "cp932", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return parse_text(text, str(p))
