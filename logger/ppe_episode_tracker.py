# ppe_episode_logger.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timezone, timedelta
import logging
import json
import os
import re

# ─────────────────────────────
# Helpers
# ─────────────────────────────
def _to_datetime(ts: Optional[float | datetime], tz: timezone) -> datetime:
    """ts may be float epoch or datetime; return tz-aware datetime."""
    if ts is None:
        return datetime.now(tz)
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=tz)
    return datetime.fromtimestamp(float(ts), tz=tz)

def _iso_date_time_parts(dt: datetime) -> Tuple[str, str]:
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # millis
    return date_str, time_str

def _pct(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v) * 100.0, 1)

def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "", s or "")

# ─────────────────────────────
# JSON Lines Formatter
# ─────────────────────────────
class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = record.msg if isinstance(record.msg, dict) else {"message": str(record.msg)}
        return json.dumps(payload, ensure_ascii=False)

def _build_logger(base_dir: str, site: str, camera_id: str, filename: str = "ppe-all.jsonl", also_console: bool = True) -> logging.Logger:
    os.makedirs(os.path.join(base_dir, site, camera_id), exist_ok=True)
    logger = logging.getLogger(f"ppe.{site}.{camera_id}")
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

# ─────────────────────────────
# Data structures
# ─────────────────────────────
@dataclass
class FrameObs:
    ts: datetime
    frame_id: int
    missing: List[str]                  # e.g. ["helmet","vest"] or []
    conf: Dict[str, float]              # 0..1 (only present keys matter)

@dataclass
class PersonState:
    """Episode state machine + periyodik emisyon bilgileri."""
    status: str = "compliant"           # compliant | pre | violation
    t0: Optional[datetime] = None       # episode start
    t_evidence: Optional[datetime] = None  # t0 + evidence_delay
    pre_buffer: deque = field(default_factory=lambda: deque(maxlen=300))
    conf_max: Dict[str, float] = field(default_factory=dict)
    evidence_frame_id: Optional[int] = None
    evidence_conf: Dict[str, float] = field(default_factory=dict)
    evidence_missing: List[str] = field(default_factory=list)
    last_violation_ts: Optional[datetime] = None
    last_seen_ts: Optional[datetime] = None

    # Periyodik kayıtlar için:
    chunk_start: Optional[datetime] = None      # bir sonraki 5sn penceresinin başlangıcı
    last_emitted_ts: Optional[datetime] = None  # en son ara kayıt zamanı
    last_frame_id: Optional[int] = None         # en son görülen frame_id
    last_missing: List[str] = field(default_factory=list)    # son frame missing
    last_conf: Dict[str, float] = field(default_factory=dict) # son frame conf

@dataclass
class PersonObservationInput:
    person_id: Optional[str]
    missing: List[str]
    conf: Dict[str, float] = field(default_factory=dict)

# ─────────────────────────────
# Main class
# ─────────────────────────────
class PPEEpisodeTracker:
    """
    Episode-based PPE violation logger.

    Video/Webcam:
      • İhlal sürerken HER 5 SANİYEDE BİR ara kayıt (duration_s = tam 5.0)
      • İhlal kapanınca tek BİTİŞ KAYDI (duration_s = toplam bölüm süresi)

    Fotoğraf:
      • Sadece 5 saniyelik aralıklarla ara kayıt (final kayıt yok)

    Not: Fotoğraf modunda 5 sn’lik kayıtları görmek için process_frame çağrıları zaman geçtikçe
    devam etmelidir veya flush() kapanışta biriken aralıkları yazacaktır.
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
        filename: str = "ppe-all.jsonl",
        mode: str = "global",                  # 'global' or 'per_person'
        include_person_id: bool = False,       # person_id logs'a yazılmaz
        source_kind: str = "video",            # 'video' | 'photo'
        periodic_log_s: Optional[float] = None # None => evidence_delay_s
    ):
        assert mode in ("global", "per_person")
        assert source_kind in ("video", "photo")
        self.site = site
        self.camera_id = camera_id
        self.tz = timezone(timedelta(hours=tz_offset_hours))
        self.evidence_delay_s = float(evidence_delay_s)
        self.periodic_log_s = float(periodic_log_s or evidence_delay_s)
        self.end_grace_s = float(end_grace_s)
        self.no_detection_timeout_s = float(no_detection_timeout_s)
        self.logger = _build_logger(log_base_dir, site, camera_id, filename, also_console)
        self.mode = mode
        self.include_person_id = include_person_id
        self.source_kind = source_kind
        self.people: Dict[str, PersonState] = {}

    # ─────────────────────────
    # Public API
    # ─────────────────────────
    def process_frame(self, frame_id: int, frame_ts: Optional[float | datetime], observations: List[PersonObservationInput | dict]) -> None:
        """
        observations: list of {person_id, missing:[...], conf:{...}} dicts or PersonObservationInput.
        For 'global' mode: merged; otherwise processed individually.
        """
        now = _to_datetime(frame_ts, self.tz)

        if self.mode == "global":
            merged_missing: List[str] = []
            merged_conf: Dict[str, float] = {}
            any_violation = False

            for obs in observations or []:
                if not isinstance(obs, PersonObservationInput):
                    obs = PersonObservationInput(
                        person_id=obs.get("person_id"),
                        missing=list(obs.get("missing", [])),
                        conf=dict(obs.get("conf", {}))
                    )
                if obs.missing:
                    any_violation = True
                for m in obs.missing:
                    if m not in merged_missing:
                        merged_missing.append(m)
                for k, v in (obs.conf or {}).items():
                    if v is None:
                        continue
                    merged_conf[k] = max(merged_conf.get(k, 0.0), float(v))

            frame_obs = FrameObs(
                ts=now,
                frame_id=int(frame_id),
                missing=sorted(merged_missing) if any_violation else [],
                conf=merged_conf
            )
            self._update_person("_GLOBAL_", frame_obs)
            return

        # per_person mode
        seen_ids = set()
        for obs in observations or []:
            if not isinstance(obs, PersonObservationInput):
                obs = PersonObservationInput(
                    person_id=obs.get("person_id"),
                    missing=list(obs.get("missing", [])),
                    conf=dict(obs.get("conf", {}))
                )
            pid = obs.person_id or "_UNKNOWN_"
            seen_ids.add(pid)
            self._update_person(pid, FrameObs(ts=now, frame_id=int(frame_id), missing=sorted(obs.missing), conf=obs.conf))

        # Absent persons
        for pid, st in list(self.people.items()):
            if pid in seen_ids:
                continue
            self._handle_absent_person(pid, now)

    def flush(self) -> None:
        """Close any open episodes gracefully (e.g., on shutdown)."""
        now = datetime.now(self.tz)
        for pid in list(self.people.keys()):
            self._handle_absent_person(pid, now, force_close=True)

    # ─────────────────────────
    # Internal logic
    # ─────────────────────────
    def _state(self, person_key: str) -> PersonState:
        if person_key not in self.people:
            self.people[person_key] = PersonState()
        return self.people[person_key]

    def _update_person(self, person_key: str, obs: FrameObs) -> None:
        st = self._state(person_key)
        st.last_seen_ts = obs.ts

        # Son gözlem bilgilerini tut (periyodik kayıtlar için)
        st.last_frame_id = obs.frame_id
        st.last_missing = list(obs.missing or [])
        st.last_conf = dict(obs.conf or {})

        if obs.missing:
            # violation path
            st.last_violation_ts = obs.ts
            # max conf güncelle
            for k, v in (obs.conf or {}).items():
                if v is None:
                    continue
                st.conf_max[k] = max(st.conf_max.get(k, 0.0), float(v))

            if st.status == "compliant":
                st.status = "pre"
                st.t0 = obs.ts
                st.t_evidence = st.t0 + timedelta(seconds=self.evidence_delay_s)
                st.pre_buffer.clear()
                st.pre_buffer.append(obs)
                # periyodik pencereleri t0'dan başlat
                st.chunk_start = st.t0
                st.last_emitted_ts = None

            elif st.status == "pre":
                st.pre_buffer.append(obs)
                # evidence anına ulaştıysak kilitle ve violation'a geç
                if st.t_evidence and obs.ts >= st.t_evidence:
                    self._lock_evidence_from_pre(st)
                    st.status = "violation"
                    # violation'a geçildi; periyodik emisyon dene
                    self._maybe_emit_interval(person_key, st, obs.ts)

            elif st.status == "violation":
                # periyodik emisyon
                self._maybe_emit_interval(person_key, st, obs.ts)

            return

        # compliant frame
        if st.status == "pre":
            self._reset_to_compliant(st)
        elif st.status == "violation":
            if st.last_violation_ts and (obs.ts - st.last_violation_ts).total_seconds() >= self.end_grace_s:
                # önce kalan periyotları kapat (varsa)
                self._maybe_emit_interval(person_key, st, st.last_violation_ts)
                # final (sadece video modunda)
                self._close_episode_and_log(person_key, st, end_ts=st.last_violation_ts)
                self._reset_to_compliant(st)

    def _handle_absent_person(self, person_key: str, now: datetime, force_close: bool = False) -> None:
        st = self._state(person_key)
        if st.status == "pre":
            if force_close or (st.last_seen_ts and (now - st.last_seen_ts).total_seconds() >= self.no_detection_timeout_s):
                self._reset_to_compliant(st)
        elif st.status == "violation":
            if force_close or (st.last_seen_ts and (now - st.last_seen_ts).total_seconds() >= self.no_detection_timeout_s):
                end_ts = st.last_violation_ts or st.last_seen_ts or now
                # kalan periyotları tamamla
                self._maybe_emit_interval(person_key, st, end_ts)
                # final (sadece video modunda)
                self._close_episode_and_log(person_key, st, end_ts=end_ts)
                self._reset_to_compliant(st)

    def _lock_evidence_from_pre(self, st: PersonState) -> None:
        assert st.t0 and st.t_evidence
        if not st.pre_buffer:
            st.evidence_frame_id = None
            st.evidence_conf = {}
            st.evidence_missing = []
            return
        target = st.t_evidence
        best = min(st.pre_buffer, key=lambda f: abs((f.ts - target).total_seconds()))
        st.evidence_frame_id = best.frame_id
        st.evidence_conf = dict(best.conf or {})
        st.evidence_missing = list(best.missing or [])
        st.pre_buffer.clear()

    # ─────────────────────────
    # Periyodik (5 sn) kayıtlar
    # ─────────────────────────
    def _maybe_emit_interval(self, person_key: str, st: PersonState, now: datetime) -> None:
        """İhlal sürerken her 'periodic_log_s' sürede bir ara kayıt yaz."""
        if st.status != "violation" or st.chunk_start is None:
            return
        period = timedelta(seconds=self.periodic_log_s)
        # Gerekirse birden fazla 5sn'lik pencere birikmiş olabilir; her seferde 1 tane yazalım
        if now - st.chunk_start >= period:
            end_ts = st.chunk_start + period
            # JSON payload (ara kayıt)
            date_str, time_str = _iso_date_time_parts(end_ts.astimezone(self.tz))
            conf_pct = {f"{k}_pct": _pct(v) for k, v in (st.last_conf or {}).items() if v is not None}
            if st.conf_max:
                conf_pct["conf_max_pct"] = {k: _pct(v) for k, v in st.conf_max.items()}

            episode_id = self._make_episode_id(st.t0 or end_ts)

            payload = {
                "date": date_str,
                "time": time_str,
                "event": "ppe_violation",
                "episode_id": episode_id,
                "missing": list(st.last_missing or []),   # o anki durumda eksik olanlar
                "frame_id": int(st.last_frame_id) if st.last_frame_id is not None else -1,
                "duration_s": round(self.periodic_log_s, 3),  # tam 5.0 sn pencere
                "conf": conf_pct,
                "site": self.site,
                "camera_id": self.camera_id,
            }
            # Foto modunda da ara kayıt yazılır; final yazımı _close_episode_and_log'ta kontrol edilir
            self.logger.info(payload)

            st.last_emitted_ts = end_ts
            st.chunk_start = end_ts  # bir sonraki pencere buradan başlar

    # ─────────────────────────
    # Final kayıt (özet)
    # ─────────────────────────
    def _close_episode_and_log(self, person_key: str, st: PersonState, end_ts: datetime) -> None:
        """Video modunda bölüm kapanışında toplam süreyi yaz. Foto modunda atlanır."""
        if self.source_kind == "photo":
            return  # foto modunda final kayıt yok

        assert st.t0 is not None, "Episode start missing"
        # Kanıt kilitli değilse best-effort
        if st.evidence_frame_id is None:
            if st.t_evidence and end_ts >= st.t_evidence and st.pre_buffer:
                self._lock_evidence_from_pre(st)
            else:
                st.evidence_frame_id = -1
                st.evidence_conf = {}
                st.evidence_missing = []

        duration_s = max(0.0, (end_ts - st.t0).total_seconds())
        date_str, time_str = _iso_date_time_parts(end_ts.astimezone(self.tz))

        conf_pct = {f"{k}_pct": _pct(v) for k, v in (st.evidence_conf or {}).items()}
        if st.conf_max:
            conf_pct["conf_max_pct"] = {k: _pct(v) for k, v in st.conf_max.items()}

        episode_id = self._make_episode_id(st.t0)

        payload = {
            "date": date_str,
            "time": time_str,
            "event": "ppe_violation",
            "episode_id": episode_id,
            # person_id intentionally omitted
            "missing": st.evidence_missing,
            "frame_id": st.evidence_frame_id,
            "duration_s": round(duration_s, 3),
            "conf": conf_pct,
            "site": self.site,
            "camera_id": self.camera_id,
        }
        self.logger.info(payload)

    def _reset_to_compliant(self, st: PersonState) -> None:
        st.status = "compliant"
        st.t0 = None
        st.t_evidence = None
        st.pre_buffer.clear()
        st.conf_max.clear()
        st.evidence_frame_id = None
        st.evidence_conf = {}
        st.evidence_missing = []
        st.last_violation_ts = None
        # periyodik alanlar
        st.chunk_start = None
        st.last_emitted_ts = None
        st.last_frame_id = None
        st.last_missing = []
        st.last_conf = {}

    def _make_episode_id(self, t0: datetime) -> str:
        sid = _slug(self.site)
        cam = _slug(self.camera_id)
        t0z = t0.astimezone(self.tz).strftime("%Y%m%dT%H%M%S%f")[:-3]
        return f"{sid}_{cam}_{t0z}"

# ─────────────────────────────
# Optional quick demo
# ─────────────────────────────
if __name__ == "__main__":
    import time
    # Video senaryosu: her 5sn ara + final
    tracker = PPEEpisodeTracker(site="warehouse-A", camera_id="cam-01",
                                mode="global", include_person_id=False,
                                also_console=True, source_kind="video",
                                evidence_delay_s=5.0)
    t0 = time.time()
    def ts(s): return t0 + s

    for s in range(0, 16):  # 0..15 sn arası ihlal
        tracker.process_frame(
            frame_id=10_000 + s,
            frame_ts=ts(s),
            observations=[{"person_id": None, "missing": ["helmet","vest"], "conf": {"not_vest": 0.66}}]
        )
        time.sleep(0.01)

    tracker.flush()
