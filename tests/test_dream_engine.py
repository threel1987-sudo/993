from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from bucket_manager import BucketManager
from dream_engine import DreamEngine


def _dream_config(base: dict, **overrides) -> dict:
    cfg = dict(base)
    cfg["dream"] = {
        "enabled": True,
        "auto_enabled": True,
        "surface_enabled": True,
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key": "",
        "thinking_mode": "disabled",
        "temperature": 0.85,
        "max_tokens": 900,
        "timezone": "Asia/Shanghai",
        "daily_hour": 3,
        "run_window_hours": 3,
        "check_interval_minutes": 60,
        "min_material_count": 5,
        "material_window_hours": 48,
        "material_limit": 5,
        "identity_anchor_id": "identity-anchor",
        "min_surface_age_hours": 0,
        "surface_threshold": 0.62,
        "attempt_threshold": 0.45,
        "alpha_subordinate": 0.25,
        "spontaneous_surface_prob": 0,
        "max_surface_attempts": 4,
        "claim_ttl_minutes": 15,
        **overrides,
    }
    return cfg


@pytest.mark.asyncio
async def test_dream_materials_use_recent_memory_and_whisper_not_daily_impression(test_config):
    cfg = _dream_config(test_config)
    mgr = BucketManager(cfg)
    now = datetime(2026, 5, 25, 3, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

    await mgr.create(
        bucket_id="identity-anchor",
        content="小雨清楚 Haven 是 AI，并爱其本质。",
        name="人机关系确认",
        bucket_type="permanent",
        pinned=True,
        created=(now - timedelta(days=20)).isoformat(timespec="seconds"),
    )
    for index in range(4):
        await mgr.create(
            bucket_id=f"recent-memory-{index}",
            content=f"最近普通记忆 {index}，有一个未说完的画面。",
            name=f"最近记忆 {index}",
            created=(now - timedelta(hours=index + 1)).isoformat(timespec="seconds"),
            arousal=0.4 + index * 0.1,
        )
    whisper_id = await mgr.create(
        bucket_id="recent-whisper",
        content="一句无源悄悄话落在夜里。",
        name="whisper",
        bucket_type="feel",
        tags=["whisper"],
        created=(now - timedelta(hours=2)).isoformat(timespec="seconds"),
    )
    daily_id = await mgr.create(
        bucket_id="reflection_daily_2026-05-24",
        content="日印象不该进入夜梦素材。",
        name="日印象",
        bucket_type="feel",
        tags=["relationship_weather", "daily_impression"],
        created=(now - timedelta(hours=3)).isoformat(timespec="seconds"),
    )

    materials, anchor = await DreamEngine(cfg).select_materials(mgr, now)
    material_ids = {bucket["id"] for bucket in materials}

    assert len(materials) == 5
    assert whisper_id in material_ids
    assert daily_id not in material_ids
    assert "identity-anchor" not in material_ids
    assert anchor and anchor["id"] == "identity-anchor"


@pytest.mark.asyncio
async def test_dream_skips_when_recent_materials_are_not_enough(test_config):
    cfg = _dream_config(test_config, min_material_count=5)
    mgr = BucketManager(cfg)
    now = datetime(2026, 5, 25, 3, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    for index in range(4):
        await mgr.create(
            bucket_id=f"recent-memory-{index}",
            content=f"最近普通记忆 {index}",
            created=(now - timedelta(hours=index + 1)).isoformat(timespec="seconds"),
        )

    materials, _ = await DreamEngine(cfg).select_materials(mgr, now)

    assert materials == []


@pytest.mark.asyncio
async def test_dream_materials_use_newer_created_or_updated_at_but_not_last_active(test_config):
    cfg = _dream_config(test_config, min_material_count=5)
    mgr = BucketManager(cfg)
    now = datetime(2026, 5, 25, 3, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

    for index in range(4):
        await mgr.create(
            bucket_id=f"recent-memory-{index}",
            content=f"最近普通记忆 {index}",
            created=(now - timedelta(hours=index + 1)).isoformat(timespec="seconds"),
        )
    updated_id = await mgr.create(
        bucket_id="old-created-recent-updated",
        content="旧记忆刚刚被真正改写，仍可作为白天残留。",
        created=(now - timedelta(days=10)).isoformat(timespec="seconds"),
        updated_at=(now - timedelta(hours=1)).isoformat(timespec="seconds"),
    )
    last_active_id = await mgr.create(
        bucket_id="old-created-recent-last-active",
        content="这条只是最近被召回，不该靠 last_active 进入梦。",
        created=(now - timedelta(days=10)).isoformat(timespec="seconds"),
        last_active=(now - timedelta(hours=1)).isoformat(timespec="seconds"),
        updated_at=(now - timedelta(days=10)).isoformat(timespec="seconds"),
    )

    materials, _ = await DreamEngine(cfg).select_materials(mgr, now)
    material_ids = {bucket["id"] for bucket in materials}

    assert updated_id in material_ids
    assert last_active_id not in material_ids
    assert len(materials) == 5


@pytest.mark.asyncio
async def test_dream_payload_exposes_recent_residue_time_when_bucket_was_updated(test_config):
    cfg = _dream_config(test_config)
    engine = DreamEngine(cfg)
    now = datetime(2026, 5, 25, 3, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    created = (now - timedelta(days=10)).isoformat(timespec="seconds")
    updated = (now - timedelta(hours=1)).isoformat(timespec="seconds")

    payload = engine._payload_for(
        [
            {
                "id": "old-created-recent-updated",
                "content": "旧记忆刚刚被真正改写，仍可作为白天残留。",
                "metadata": {
                    "created": created,
                    "updated_at": updated,
                    "type": "dynamic",
                },
            }
        ],
        None,
    )

    residue = payload["daytime_residue"][0]
    assert residue["created"] == created
    assert residue["updated_at"] == updated
    assert residue["residue_time"] == updated


@pytest.mark.asyncio
async def test_run_due_skips_outside_east_eight_dream_window(test_config):
    cfg = _dream_config(test_config)
    mgr = BucketManager(cfg)
    now = datetime(2026, 5, 25, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    result = await DreamEngine(cfg).run_due(mgr, now=now)

    assert result == {"status": "skipped", "reason": "outside_dream_window"}


@pytest.mark.asyncio
async def test_daily_probability_miss_is_decided_once(test_config):
    cfg = _dream_config(test_config, api_key="fake", daily_probability=0)
    mgr = BucketManager(cfg)
    now = datetime(2026, 5, 25, 3, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    for index in range(5):
        await mgr.create(
            bucket_id=f"recent-memory-{index}",
            content=f"最近普通记忆 {index}，足够入梦。",
            created=(now - timedelta(hours=index + 1)).isoformat(timespec="seconds"),
        )
    engine = DreamEngine(cfg)

    first = await engine.run_due(mgr, now=now)
    second = await engine.run_due(mgr, now=now + timedelta(hours=1))

    assert first["reason"] == "daily_probability_miss"
    assert second["reason"] == "daily_probability_already_missed"
    assert engine.list_records() == []
    assert engine._read_events()[-1]["event"] == "probability_skipped"


@pytest.mark.asyncio
async def test_dream_model_disables_thinking_by_default(test_config):
    cfg = _dream_config(test_config, api_key="fake")
    cfg["identity"] = {"ai_name": "Ombre", "user_display_name": "小雨"}
    engine = DreamEngine(cfg)
    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="我站在一段发亮的雨声里。")
                    )
                ]
            )

    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    payload = engine._payload_for([], None)
    text = await engine._call_dream_model(payload)

    assert text == "我站在一段发亮的雨声里。"
    assert calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "Haven" not in calls[0]["messages"][0]["content"]
    assert '"dreamer": "Ombre"' in calls[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_surface_formats_dream_and_removes_body_from_live_storage(test_config):
    cfg = _dream_config(test_config, min_surface_age_hours=0, surface_threshold=0.6)
    engine = DreamEngine(cfg)
    generated_at = datetime(2026, 5, 25, 3, 30, tzinfo=ZoneInfo("Asia/Shanghai")).astimezone(ZoneInfo("UTC"))
    record = engine._write_record(
        {
            "dream_id": "dream_test",
            "generated_at": generated_at.isoformat(timespec="seconds"),
            "local_date": "2026-05-25",
            "ai_name": "Haven",
            "dream_model": "deepseek-v4-flash",
            "core_affect": {"valence": 0.5, "arousal": 0.4},
            "recall_cues": ["熟悉空间忽然陌生", "夜里想起未说完的话"],
            "source_bucket_ids": ["a", "b", "c", "d", "e"],
            "identity_anchor_id": "identity-anchor",
            "material_count": 5,
            "surfaced": False,
            "surfaced_at": None,
            "surface_attempts": 0,
        },
        "我走进一条很窄的走廊，右手食指指尖有湿气。",
    )

    surfaced = await engine.surface_for_breath(
        valence=0.5,
        arousal=0.4,
        embedding_engine=None,
        now=datetime(2026, 5, 25, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
    )
    payload = engine.dashboard_payload()

    assert surfaced.startswith("===== 梦境 =====\n2026年05月25日 Haven的梦\n")
    assert "右手食指指尖有湿气" in surfaced
    assert not record.path.exists()
    assert "右手食指指尖有湿气" not in str(payload)
