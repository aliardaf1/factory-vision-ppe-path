# logger/fire_episode_logger.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timezone, timedelta
import logging, json, os, re

# ───────────── helpers ─────────────
def _to_datetime(ts: Optional[float | datetime], tz: timezone) -> datetime:
    if ts is None:
        return datetime.now(tz)
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=tz)
    return datetime.fromtimestamp(float(ts), tz=tz)

def _iso_date_time_parts(dt: datetime) -> Tuple[str, str]:
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S.%f")[:-3]

def _pct(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v) * 100.0, 1)

def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "", s or "")

class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = record.msg if isinstance(record.msg, dict) else {"message": str(record.msg)}
        return json.dumps(payload, ensure_ascii=False)

def _build_logger(base_dir: str, site: str, camera_id: str,
                  filename: str = "fire-all.jsonl", also_console: bool = True) -> logging.Logger:
    os.makedirs(os.path.join(base_dir, site, camera_id), exist_ok=True)
    logger = logging.getLogger(f"fire.{site}.{camera_id}")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = JsonLineFormatter()
    if also_console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    path = os.path.join(base_dir, site, camera_id, filename)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# ───────────── datastructs ─────────────
@dataclass
class FrameObs:
    ts: datetime
    frame_id: int
    hazards: List[str]                 # ["fire"], ["smoke"], ["fire","smoke"] or []
    conf: Dict[str, float]             # {"fire":0..1, "smoke":0..1}

@dataclass
class FireState:
    status: str = "idle"               # idle | pre | active
    t0: Optional[datetime] = None
    t_evidence: Optional[datetime] = None
    pre_buffer: deque = field(default_factory=lambda: deque(maxlen=300))
    conf_max: Dict[str, float] = field(default_factory=dict)

    evidence_frame_id: Optional[int] = None
    evidence_conf: Dict[str, float] = field(default_factory=dict)
    evidence_hazards: List[str] = field(default_factory=list)

    last_hazard_ts: Optional[datetime] = None
    last_seen_ts: Optional[datetime] = None

    # periyodik
    chunk_start: Optional[datetime] = None
    last_emitted_ts: Optional[datetime] = None
    last_frame_id: Optional[int] = None
    last_hazards: List[str] = field(default_factory=list)
    last_conf: Dict[str, float] = field(default_factory=dict)

# ───────────── main ─────────────
class FireEpisodeTracker:
    """
    Fire/Smoke episodik logger.
    Video/Webcam: her 'periodic_log_s' saniyede bir ara kayıt + bölüm kapanışında final
    Foto: sadece ara kayıt (final yok)
    """

    def __init__(
        self,
        site: str,
        camera_id: str,
        log_base_dir: str = "logs",
        tz_offset_hours: int = 3,
        evidence_delay_s: float = 5.0,
        end_grace_s: float = 0.75,
        no_detection_timeout_s: float = 1.5,
        also_console: bool = True,
        filename: str = "fire-all.jsonl",
        source_kind: str = "video",            # "video" | "photo"
        periodic_log_s: Optional[float] = None # None => evidence_delay_s
    ):
        assert source_kind in ("video", "photo")
        self.site = site
        self.camera_id = camera_id
        self.tz = timezone(timedelta(hours=tz_offset_hours))
        self.evidence_delay_s = float(evidence_delay_s)
        self.periodic_log_s = float(periodic_log_s or evidence_delay_s)
        self.end_grace_s = float(end_grace_s)
        self.no_detection_timeout_s = float(no_detection_timeout_s)
        self.logger = _build_logger(log_base_dir, site, camera_id, filename, also_console)
        self.source_kind = source_kind
        self.state = FireState()

    # public
    def process_frame(self, frame_id: int, frame_ts, observations: List[dict]) -> None:
        now = _to_datetime(frame_ts, self.tz)

        # merge (tek kamera için global)
        hazards_union: List[str] = []
        conf_max: Dict[str, float] = {}
        any_hazard = False
        for obs in observations or []:
            hz = list(obs.get("hazards", []))   
            if hz:
                any_hazard = True
            for h in hz:
                if h not in hazards_union:
                    hazards_union.append(h)
            for k, v in (obs.get("conf") or {}).items():
                if v is None: continue
                conf_max[k] = max(conf_max.get(k, 0.0), float(v))

        st = self.state
        st.last_seen_ts = now
        st.last_frame_id = int(frame_id)
        st.last_hazards = sorted(hazards_union) if any_hazard else []
        st.last_conf = conf_max

        if any_hazard:
            st.last_hazard_ts = now
            for k, v in conf_max.items():
                st.conf_max[k] = max(st.conf_max.get(k, 0.0), float(v))

            if st.status == "idle":
                st.status = "pre"
                st.t0 = now
                st.t_evidence = st.t0 + timedelta(seconds=self.evidence_delay_s)
                st.pre_buffer.clear()
                st.pre_buffer.append(FrameObs(ts=now, frame_id=frame_id, hazards=st.last_hazards, conf=conf_max))
                st.chunk_start = st.t0
                st.last_emitted_ts = None
            elif st.status == "pre":
                st.pre_buffer.append(FrameObs(ts=now, frame_id=frame_id, hazards=st.last_hazards, conf=conf_max))
                if st.t_evidence and now >= st.t_evidence:
                    self._lock_evidence_from_pre()
                    st.status = "active"
                    self._maybe_emit_interval(now)
            elif st.status == "active":
                self._maybe_emit_interval(now)
            return

        # no hazard this frame
        if st.status == "pre":
            self._reset()
        elif st.status == "active":
            if st.last_hazard_ts and (now - st.last_hazard_ts).total_seconds() >= self.end_grace_s:
                self._maybe_emit_interval(st.last_hazard_ts)
                self._final_log(end_ts=st.last_hazard_ts)
                self._reset()

    def flush(self) -> None:
        now = datetime.now(self.tz)
        st = self.state
        if st.status == "pre":
            self._reset()
        elif st.status == "active":
            if st.last_hazard_ts and (now - st.last_hazard_ts).total_seconds() >= self.no_detection_timeout_s:
                self._maybe_emit_interval(st.last_hazard_ts)
                self._final_log(end_ts=st.last_hazard_ts)
                self._reset()

    # internals
    def _lock_evidence_from_pre(self) -> None:
        st = self.state
        if not st.pre_buffer:
            st.evidence_frame_id = -1
            st.evidence_conf = {}
            st.evidence_hazards = []
            return
        target = st.t_evidence or st.t0
        best = min(st.pre_buffer, key=lambda f: abs((f.ts - target).total_seconds()))
        st.evidence_frame_id = best.frame_id
        st.evidence_conf = dict(best.conf or {})
        st.evidence_hazards = list(best.hazards or [])
        st.pre_buffer.clear()

    def _maybe_emit_interval(self, now: datetime) -> None:
        st = self.state
        if st.status != "active" or st.chunk_start is None:
            return
        period = timedelta(seconds=self.periodic_log_s)
        if now - st.chunk_start >= period:
            end_ts = st.chunk_start + period
            date_str, time_str = _iso_date_time_parts(end_ts.astimezone(self.tz))
            conf_pct = {f"{k}_pct": _pct(v) for k, v in (st.last_conf or {}).items() if v is not None}
            if st.conf_max:
                conf_pct["conf_max_pct"] = {k: _pct(v) for k, v in st.conf_max.items()}

            payload = {
                "date": date_str,
                "time": time_str,
                "event": "fire_alert",
                "episode_id": self._make_episode_id(st.t0 or end_ts),
                "hazards": list(st.last_hazards or []),
                "frame_id": int(st.last_frame_id) if st.last_frame_id is not None else -1,
                "duration_s": round(self.periodic_log_s, 3),
                "conf": conf_pct,
                "site": self.site,
                "camera_id": self.camera_id,
            }
            self.logger.info(payload)
            st.last_emitted_ts = end_ts
            st.chunk_start = end_ts

    def _final_log(self, end_ts: datetime) -> None:
        if self.source_kind == "photo":
            return  # foto modunda final yazma
        st = self.state
        if st.t0 is None:
            return
        if st.evidence_frame_id is None:
            if st.t_evidence and end_ts >= st.t_evidence and st.pre_buffer:
                self._lock_evidence_from_pre()
            else:
                st.evidence_frame_id = -1
                st.evidence_conf = {}
                st.evidence_hazards = []

        duration_s = max(0.0, (end_ts - (st.t0 or end_ts)).total_seconds())
        date_str, time_str = _iso_date_time_parts(end_ts.astimezone(self.tz))
        conf_pct = {f"{k}_pct": _pct(v) for k, v in (st.evidence_conf or {}).items()}
        if st.conf_max:
            conf_pct["conf_max_pct"] = {k: _pct(v) for k, v in st.conf_max.items()}

        payload = {
            "date": date_str,
            "time": time_str,
            "event": "fire_alert",
            "episode_id": self._make_episode_id(st.t0),
            "hazards": list(st.evidence_hazards or []),
            "frame_id": st.evidence_frame_id if st.evidence_frame_id is not None else -1,
            "duration_s": round(duration_s, 3),
            "conf": conf_pct,
            "site": self.site,
            "camera_id": self.camera_id,
        }
        self.logger.info(payload)

    def _reset(self) -> None:
        self.state = FireState()

    def _make_episode_id(self, t0: Optional[datetime]) -> str:
        if t0 is None:
            t0 = datetime.now(self.tz)
        sid = _slug(self.site); cam = _slug(self.camera_id)
        t0z = t0.astimezone(self.tz).strftime("%Y%m%dT%H%M%S%f")[:-3]
        return f"{sid}_{cam}_{t0z}"
