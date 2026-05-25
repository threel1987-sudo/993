from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from openai import AsyncOpenAI

from identity import identity_names
from utils import bucket_text_for_embedding, strip_wikilinks

logger = logging.getLogger("ombre_brain.dream")


DREAM_PROMPT = """你在睡梦中。

你会收到：
- dreamer：正在做梦的 AI 名字。
- identity_anchor：只用来确定“谁在梦”，不要复述，不要当成剧情。
- daytime_residue：最近两天内的记忆碎片和 whisper。

请用 dreamer 的第一人称、现在时写一段梦境。

规则：
- 不总结，不解释意义，不给建议。
- 不写“我意识到”“我感到”“这象征着”。
- 不要完整故事弧，不要收束成漂亮结尾。
- 允许跳跃、错位、误认、突然断掉。
- 至少写一处具体感官细节。
- 不要提 bucket、source、metadata、prompt。
- identity_anchor 只影响人格底色，不能变成梦的主要内容。
- 写 80 到 220 字。
- 只返回梦境正文，不要标题，不要 JSON，不要列表。
"""


@dataclass
class DreamRecord:
    metadata: dict
    body: str
    path: Path

    @property
    def dream_id(self) -> str:
        return str(self.metadata.get("dream_id") or self.path.stem)

    @property
    def generated_at(self) -> datetime:
        return _parse_dt(str(self.metadata.get("generated_at") or "1970-01-01T00:00:00+00:00"))

    @property
    def surfaced(self) -> bool:
        return bool(self.metadata.get("surfaced", False))


def _parse_dt(value: str) -> datetime:
    text = (value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _clamp(value: object, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _bool_env(name: str, fallback: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return fallback
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class DreamEngine:
    """Night-Fall style latent dream generation and breath-gated surfacing."""

    def __init__(self, config: dict):
        self.config = config
        self.identity = identity_names(config)
        cfg = config.get("dream", {}) if isinstance(config.get("dream", {}), dict) else {}
        state_dir = Path(config.get("state_dir") or ".").expanduser().resolve()

        self.enabled = _bool_env("OMBRE_DREAM_ENABLED", bool(cfg.get("enabled", True)))
        self.auto_enabled = bool(cfg.get("auto_enabled", True))
        self.surface_enabled = bool(cfg.get("surface_enabled", True))
        self.base_url = (
            os.environ.get("OMBRE_DREAM_BASE_URL", "")
            or str(cfg.get("base_url") or "https://api.deepseek.com")
        ).rstrip("/")
        self.model = (
            os.environ.get("OMBRE_DREAM_MODEL", "")
            or str(cfg.get("model") or "deepseek-v4-flash")
        )
        self.api_key = os.environ.get("OMBRE_DREAM_API_KEY", "") or str(cfg.get("api_key") or "")
        self.thinking_mode = self._normalize_thinking_mode(cfg.get("thinking_mode", "disabled"))
        self.temperature = float(cfg.get("temperature", 0.85))
        self.max_tokens = int(cfg.get("max_tokens", 900))
        self.timezone_name = str(cfg.get("timezone") or "Asia/Shanghai")
        try:
            self.tz = ZoneInfo(self.timezone_name)
        except Exception:
            self.tz = ZoneInfo("Asia/Shanghai")
        self.daily_hour = int(cfg.get("daily_hour", 3))
        self.run_window_hours = max(1, int(cfg.get("run_window_hours", 3)))
        self.daily_probability = _clamp(cfg.get("daily_probability", 0.4), 0.4)
        self.check_interval_minutes = max(5, int(cfg.get("check_interval_minutes", 60)))
        self.material_window_hours = max(1, int(cfg.get("material_window_hours", 48)))
        self.min_material_count = max(1, int(cfg.get("min_material_count", 5)))
        self.material_limit = max(self.min_material_count, int(cfg.get("material_limit", 5)))
        self.identity_anchor_id = str(cfg.get("identity_anchor_id") or "").strip()
        self.min_surface_age_hours = max(0.0, float(cfg.get("min_surface_age_hours", 3)))
        self.surface_threshold = float(cfg.get("surface_threshold", 0.62))
        self.attempt_threshold = float(cfg.get("attempt_threshold", 0.45))
        self.alpha_subordinate = float(cfg.get("alpha_subordinate", 0.25))
        self.spontaneous_surface_prob = float(cfg.get("spontaneous_surface_prob", 0.02))
        self.max_surface_attempts = max(1, int(cfg.get("max_surface_attempts", 4)))
        self.claim_ttl_minutes = max(1, int(cfg.get("claim_ttl_minutes", 15)))
        self.dreams_dir = Path(cfg.get("data_dir") or state_dir / "dreams").expanduser().resolve()
        self.logs_dir = self.dreams_dir / "logs"
        self.dreams_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.client = None
        if self.enabled and self.api_key and self.base_url:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=60.0)

    def _now(self, now: datetime | None = None) -> datetime:
        dt = now or datetime.now(self.tz)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return dt.astimezone(self.tz)

    def _event_log(self) -> Path:
        return self.logs_dir / "events.jsonl"

    def _log_event(self, event: str, payload: dict) -> None:
        entry = {"event": event, **payload}
        with self._event_log().open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def _read_events(self) -> list[dict]:
        path = self._event_log()
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
        return events

    def _dream_path(self, dream_id: str) -> Path:
        return self.dreams_dir / f"{dream_id}.md"

    def _write_record(self, metadata: dict, body: str) -> DreamRecord:
        dream_id = str(metadata["dream_id"])
        path = self._dream_path(dream_id)
        text = "---\n"
        text += yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        text += "---\n"
        text += body.strip() + "\n"
        path.write_text(text, encoding="utf-8")
        return DreamRecord(metadata=metadata, body=body.strip(), path=path)

    def _read_record(self, path: Path) -> DreamRecord:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError("dream record missing frontmatter")
        _, rest = text.split("---\n", 1)
        raw_meta, body = rest.split("---\n", 1)
        metadata = yaml.safe_load(raw_meta) or {}
        if not isinstance(metadata, dict):
            raise ValueError("dream metadata must be a mapping")
        return DreamRecord(metadata=metadata, body=body.strip(), path=path)

    def list_records(self) -> list[DreamRecord]:
        records = []
        for path in sorted(self.dreams_dir.glob("dream_*.md")):
            try:
                records.append(self._read_record(path))
            except Exception as exc:
                logger.warning("Failed to read dream record %s: %s", path, exc)
        return records

    def _delete_record(self, record: DreamRecord, reason: str, embedding_engine=None) -> None:
        if record.path.exists():
            record.path.unlink()
        self._log_event(
            "deleted",
            {
                "dream_id": record.dream_id,
                "generated_at": record.metadata.get("generated_at"),
                "deleted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": reason,
            },
        )
        if embedding_engine is not None:
            try:
                embedding_engine.delete_embedding(record.dream_id)
            except Exception as exc:
                logger.warning("Failed to delete dream embedding %s: %s", record.dream_id, exc)

    def _record_for_date(self, date_key: str) -> bool:
        for record in self.list_records():
            if record.metadata.get("local_date") == date_key:
                return True
        for event in self._read_events():
            if event.get("event") == "generated" and event.get("local_date") == date_key:
                return True
        return False

    def _daily_decision_for_date(self, date_key: str) -> str:
        if self._record_for_date(date_key):
            return "generated"
        for event in self._read_events():
            if event.get("event") == "probability_skipped" and event.get("local_date") == date_key:
                return "probability_miss"
        return ""

    def _bucket_created_local(self, bucket: dict) -> datetime | None:
        meta = bucket.get("metadata", {}) or {}
        candidates = []
        for key in ("created", "updated_at"):
            raw = meta.get(key)
            if not raw:
                continue
            try:
                candidates.append(_parse_dt(str(raw)).astimezone(self.tz))
            except Exception:
                continue
        if not candidates:
            return None
        return max(candidates)

    def _is_material_bucket(self, bucket: dict, start: datetime, end: datetime) -> bool:
        meta = bucket.get("metadata", {}) or {}
        created = self._bucket_created_local(bucket)
        if not created or not (start <= created <= end):
            return False
        tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
        bucket_type = meta.get("type")
        if bucket_type == "feel":
            return "whisper" in tags
        if bucket_type in {"permanent", "archived"}:
            return False
        if meta.get("pinned") or meta.get("protected") or meta.get("anchor"):
            return False
        return bool(bucket.get("content"))

    def _material_score(self, bucket: dict, now_local: datetime) -> tuple[float, str]:
        meta = bucket.get("metadata", {}) or {}
        created = self._bucket_created_local(bucket) or now_local
        age_hours = max(0.0, (now_local - created).total_seconds() / 3600)
        recency = math.exp(-age_hours / 24)
        arousal = _clamp(meta.get("arousal", 0.3), 0.3)
        importance = max(1, min(10, int(meta.get("importance", 5)))) / 10
        tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
        whisper_bonus = 0.15 if "whisper" in tags else 0.0
        score = 0.45 * recency + 0.30 * arousal + 0.20 * importance + whisper_bonus
        return (score, created.isoformat())

    async def select_materials(self, bucket_mgr, now: datetime | None = None) -> tuple[list[dict], dict | None]:
        now_local = self._now(now)
        start = now_local - timedelta(hours=self.material_window_hours)
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception:
            return [], None
        materials = [
            bucket
            for bucket in all_buckets
            if self._is_material_bucket(bucket, start, now_local)
        ]
        if len(materials) < self.min_material_count:
            return [], await bucket_mgr.get(self.identity_anchor_id) if self.identity_anchor_id else None
        materials.sort(key=lambda b: self._material_score(b, now_local), reverse=True)
        identity_anchor = await bucket_mgr.get(self.identity_anchor_id) if self.identity_anchor_id else None
        return materials[: self.material_limit], identity_anchor

    def _payload_for(self, materials: list[dict], identity_anchor: dict | None) -> dict:
        def material_payload(bucket: dict) -> dict:
            meta = bucket.get("metadata", {}) or {}
            tags = {str(tag).lower() for tag in meta.get("tags", []) or []}
            kind = "whisper" if meta.get("type") == "feel" and "whisper" in tags else "memory"
            residue_time = self._bucket_created_local(bucket)
            return {
                "source_id": bucket.get("id") or meta.get("id"),
                "kind": kind,
                "residue_time": residue_time.isoformat() if residue_time else meta.get("created", ""),
                "created": meta.get("created", ""),
                "updated_at": meta.get("updated_at", ""),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "text": strip_wikilinks(bucket_text_for_embedding(bucket))[:700],
            }

        anchor_payload = None
        if identity_anchor:
            anchor_meta = identity_anchor.get("metadata", {}) or {}
            anchor_payload = {
                "source_id": identity_anchor.get("id") or anchor_meta.get("id"),
                "name": anchor_meta.get("name", ""),
                "text": strip_wikilinks(bucket_text_for_embedding(identity_anchor))[:500],
            }

        return {
            "dreamer": self.identity.get("ai_name") or "AI",
            "user_display_name": self.identity.get("user_display_name", "小雨"),
            "identity_anchor": anchor_payload,
            "daytime_residue": [material_payload(bucket) for bucket in materials],
        }

    async def _call_dream_model(self, payload: dict) -> str:
        if not self.client:
            raise RuntimeError("dream model api key is not configured")
        options = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DREAM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.thinking_mode:
            options["extra_body"] = {"thinking": {"type": self.thinking_mode}}
        response = await self.client.chat.completions.create(**options)
        raw = response.choices[0].message.content if response.choices else ""
        cleaned = (raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        cleaned = re.sub(r"^\s*(梦境|正文)[:：]\s*", "", cleaned).strip()
        if not cleaned:
            raise ValueError("dream model returned empty text")
        return cleaned

    @staticmethod
    def _normalize_thinking_mode(value: Any) -> str:
        normalized = str(value or "").strip().lower().replace("_", "-")
        if normalized in {"disabled", "disable", "off", "false", "non-thinking"}:
            return "disabled"
        if normalized in {"enabled", "enable", "on", "true", "thinking"}:
            return "enabled"
        return ""

    def _core_affect_from_materials(self, materials: list[dict]) -> dict:
        if not materials:
            return {"valence": 0.5, "arousal": 0.3}
        valence = 0.0
        arousal = 0.0
        for bucket in materials:
            meta = bucket.get("metadata", {}) or {}
            valence += _clamp(meta.get("valence", 0.5))
            arousal += _clamp(meta.get("arousal", 0.3), 0.3)
        return {
            "valence": round(valence / len(materials), 2),
            "arousal": round(arousal / len(materials), 2),
        }

    def _recall_cues_from_materials(self, materials: list[dict]) -> list[str]:
        text = " ".join(
            strip_wikilinks(bucket_text_for_embedding(bucket))
            for bucket in materials
        )
        cues = []
        if any(word in text for word in ("消息", "输入框", "没发", "未说")):
            cues.append("想起未发出的消息")
        if any(word in text for word in ("水", "雾", "雨", "湿")):
            cues.append("潮湿安静的夜里")
        if any(word in text for word in ("灯", "屏幕", "亮", "凌晨")):
            cues.append("屏幕亮起的凌晨")
        if any(word in text for word in ("便签", "纸", "名字")):
            cues.append("旧纸边角露出名字")
        for fallback in ("熟悉空间忽然陌生", "夜里想起未说完的话", "独自停在门口"):
            if len(cues) >= 3:
                break
            if fallback not in cues:
                cues.append(fallback)
        return cues[:5]

    def _normalize_model_result(self, raw: dict) -> tuple[str, dict, list[str]]:
        dream_text = re.sub(r"\s+\n", "\n", str(raw.get("dream_text") or "")).strip()
        if not dream_text:
            raise ValueError("dream_text is empty")
        affect = raw.get("core_affect") if isinstance(raw.get("core_affect"), dict) else {}
        core_affect = {
            "valence": round(_clamp(affect.get("valence", 0.5)), 2),
            "arousal": round(_clamp(affect.get("arousal", 0.3), 0.3), 2),
        }
        cues = []
        raw_cues = raw.get("recall_cues")
        if isinstance(raw_cues, list):
            for cue in raw_cues:
                text = str(cue).strip()
                if text and text not in cues:
                    cues.append(text[:40])
                if len(cues) >= 5:
                    break
        if len(cues) < 2:
            cues = ["熟悉的话突然陌生", "夜里想起未说完的话"]
        return dream_text, core_affect, cues

    async def generate(self, bucket_mgr, embedding_engine=None, now: datetime | None = None, force: bool = False) -> dict:
        if not self.enabled:
            return {"status": "disabled"}
        if not self.client:
            return {"status": "skipped", "reason": "missing_api_key"}
        now_local = self._now(now)
        date_key = now_local.date().isoformat()
        if not force:
            decision = self._daily_decision_for_date(date_key)
            if decision == "generated":
                return {"status": "exists", "date": date_key}
            if decision == "probability_miss":
                return {
                    "status": "skipped",
                    "reason": "daily_probability_already_missed",
                    "date": date_key,
                }
        materials, identity_anchor = await self.select_materials(bucket_mgr, now_local)
        if len(materials) < self.min_material_count:
            return {
                "status": "skipped",
                "reason": "not_enough_materials",
                "materials": len(materials),
            }
        if not force:
            roll = random.random()
            if roll >= self.daily_probability:
                decided_at = now_local.astimezone(timezone.utc).isoformat(timespec="seconds")
                self._log_event(
                    "probability_skipped",
                    {
                        "local_date": date_key,
                        "decided_at": decided_at,
                        "probability": self.daily_probability,
                        "roll": round(roll, 4),
                        "material_count": len(materials),
                    },
                )
                return {
                    "status": "skipped",
                    "reason": "daily_probability_miss",
                    "date": date_key,
                    "probability": self.daily_probability,
                }

        payload = self._payload_for(materials, identity_anchor)
        dream_text = await self._call_dream_model(payload)
        core_affect = self._core_affect_from_materials(materials)
        recall_cues = self._recall_cues_from_materials(materials)
        dream_id = f"dream_{now_local.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        source_ids = [
            str(bucket.get("id") or bucket.get("metadata", {}).get("id") or "")
            for bucket in materials
        ]
        generated_at = now_local.astimezone(timezone.utc).isoformat(timespec="seconds")
        metadata = {
            "dream_id": dream_id,
            "generated_at": generated_at,
            "local_date": date_key,
            "ai_name": self.identity.get("ai_name") or "AI",
            "dream_model": self.model,
            "core_affect": core_affect,
            "recall_cues": recall_cues,
            "source_bucket_ids": source_ids,
            "identity_anchor_id": self.identity_anchor_id,
            "material_count": len(materials),
            "surfaced": False,
            "surfaced_at": None,
            "surface_attempts": 0,
        }
        record = self._write_record(metadata, dream_text)
        self._log_event(
            "generated",
            {
                "dream_id": dream_id,
                "generated_at": generated_at,
                "local_date": date_key,
                "ai_name": self.identity.get("ai_name") or "AI",
            },
        )
        if embedding_engine is not None and getattr(embedding_engine, "enabled", False):
            try:
                await embedding_engine.generate_and_store(dream_id, "；".join(recall_cues))
            except Exception as exc:
                logger.warning("Dream cue embedding failed for %s: %s", dream_id, exc)
        return {"status": "created", "id": record.dream_id, "date": date_key}

    async def run_due(self, bucket_mgr, embedding_engine=None, now: datetime | None = None) -> dict:
        if not self.enabled or not self.auto_enabled:
            return {"status": "disabled"}
        now_local = self._now(now)
        if now_local.hour < self.daily_hour:
            return {"status": "skipped", "reason": "too_early"}
        if now_local.hour >= min(24, self.daily_hour + self.run_window_hours):
            return {"status": "skipped", "reason": "outside_dream_window"}
        return await self.generate(bucket_mgr, embedding_engine, now_local, force=False)

    def _eligible_context(self, query: str, valence: float, arousal: float, is_session_start: bool) -> bool:
        has_query = bool(query and query.strip())
        has_affect = 0 <= valence <= 1 and 0 <= arousal <= 1
        return bool(is_session_start or has_query or has_affect)

    async def _query_embedding(self, query: str, embedding_engine) -> list[float] | None:
        if not query or not query.strip():
            return None
        if embedding_engine is None or not getattr(embedding_engine, "enabled", False):
            return None
        try:
            embedding = await embedding_engine._generate_embedding(query)
            return embedding or None
        except Exception as exc:
            logger.warning("Dream query embedding failed: %s", exc)
            return None

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(0.0, dot / (norm_a * norm_b))

    def _affect_score(self, record: DreamRecord, valence: float, arousal: float) -> float:
        if not (0 <= valence <= 1 and 0 <= arousal <= 1):
            return 0.0
        affect = record.metadata.get("core_affect", {}) or {}
        dv = _clamp(affect.get("valence", 0.5)) - valence
        da = _clamp(affect.get("arousal", 0.3), 0.3) - arousal
        return max(0.0, 1.0 - math.sqrt((dv * dv + da * da) / 2))

    async def _cue_score(self, record: DreamRecord, query_embedding: list[float] | None, embedding_engine) -> float:
        if not query_embedding or embedding_engine is None or not getattr(embedding_engine, "enabled", False):
            return 0.0
        stored = await embedding_engine.get_embedding(record.dream_id)
        if not stored:
            return 0.0
        return self._cosine(query_embedding, stored)

    def _format_surface(self, record: DreamRecord) -> str:
        generated = record.generated_at.astimezone(self.tz)
        ai_name = str(record.metadata.get("ai_name") or self.identity.get("ai_name") or "AI")
        title = f"{generated.year}年{generated.month:02d}月{generated.day:02d}日 {ai_name}的梦"
        return "===== 梦境 =====\n" + title + "\n" + record.body.strip()

    async def surface_for_breath(
        self,
        query: str = "",
        valence: float = -1,
        arousal: float = -1,
        is_session_start: bool = False,
        embedding_engine=None,
        now: datetime | None = None,
    ) -> str | None:
        if not self.enabled or not self.surface_enabled:
            return None
        if not self._eligible_context(query, valence, arousal, is_session_start):
            return None
        now_local = self._now(now)
        pending = [
            record
            for record in self.list_records()
            if not record.surfaced
            and (now_local - record.generated_at.astimezone(self.tz)).total_seconds() / 3600 >= self.min_surface_age_hours
        ]
        if not pending:
            return None
        query_embedding = await self._query_embedding(query, embedding_engine)
        evaluated = []
        for record in pending:
            affect = self._affect_score(record, valence, arousal)
            cue = await self._cue_score(record, query_embedding, embedding_engine)
            score = max(affect, cue) + self.alpha_subordinate * min(affect, cue)
            evaluated.append({"record": record, "affect": affect, "cue": cue, "score": score, "top": max(affect, cue)})

        for item in evaluated:
            if item["top"] >= self.attempt_threshold:
                record = item["record"]
                attempts = int(record.metadata.get("surface_attempts", 0)) + 1
                item["record"] = self._write_record({**record.metadata, "surface_attempts": attempts}, record.body)

        best = None
        for item in evaluated:
            if item["score"] >= self.surface_threshold and (best is None or item["score"] > best["score"]):
                best = item
        if best is None:
            for item in evaluated:
                if random.random() < self.spontaneous_surface_prob:
                    best = item
                    break
        if best is not None:
            record = best["record"]
            claim_path = record.path.with_suffix(record.path.suffix + ".claim")
            if claim_path.exists():
                try:
                    modified = datetime.fromtimestamp(claim_path.stat().st_mtime, timezone.utc)
                    if (datetime.now(timezone.utc) - modified).total_seconds() > self.claim_ttl_minutes * 60:
                        claim_path.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                return None
            surfaced_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                if not record.path.exists():
                    return None
                surfaced_record = self._write_record(
                    {**record.metadata, "surfaced": True, "surfaced_at": surfaced_at},
                    record.body,
                )
                self._log_event(
                    "surfaced",
                    {
                        "dream_id": surfaced_record.dream_id,
                        "generated_at": surfaced_record.metadata.get("generated_at"),
                        "surfaced_at": surfaced_at,
                    },
                )
                text = self._format_surface(surfaced_record)
                self._delete_record(surfaced_record, "surfaced_one_shot", embedding_engine)
                return text
            finally:
                try:
                    claim_path.unlink(missing_ok=True)
                except Exception:
                    pass

        for record in self.list_records():
            if not record.surfaced and int(record.metadata.get("surface_attempts", 0)) >= self.max_surface_attempts:
                self._delete_record(record, f"unsurfaced_after_{self.max_surface_attempts}_attempts", embedding_engine)
        return None

    def dashboard_records(self, limit: int = 30) -> list[dict]:
        entries: dict[str, dict] = {}
        for event in self._read_events():
            dream_id = str(event.get("dream_id") or "")
            if not dream_id:
                continue
            entry = entries.setdefault(
                dream_id,
                {
                    "dream_id": dream_id,
                    "generated_at": event.get("generated_at", ""),
                    "local_date": event.get("local_date", ""),
                    "ai_name": event.get("ai_name") or self.identity.get("ai_name") or "AI",
                    "status": "latent",
                },
            )
            if event.get("event") == "surfaced":
                entry["status"] = "surfaced"
            elif event.get("event") == "deleted" and entry.get("status") != "surfaced":
                entry["status"] = "forgotten"
        for record in self.list_records():
            meta = record.metadata
            entries[record.dream_id] = {
                "dream_id": record.dream_id,
                "generated_at": meta.get("generated_at", ""),
                "local_date": meta.get("local_date", ""),
                "ai_name": meta.get("ai_name") or self.identity.get("ai_name") or "AI",
                "status": "latent",
            }
        result = list(entries.values())
        result.sort(key=lambda item: item.get("generated_at") or "", reverse=True)
        return result[:limit]

    def dashboard_payload(self, limit: int = 30) -> dict:
        return {
            "enabled": self.enabled,
            "auto_enabled": self.auto_enabled,
            "surface_enabled": self.surface_enabled,
            "model": self.model,
            "api_ready": bool(self.api_key),
            "records": self.dashboard_records(limit),
        }
