import asyncio
import importlib.util
import sys
from pathlib import Path


def _load_cleanup_module():
    path = Path("scripts/cleanup_duplicate_buckets.py")
    spec = importlib.util.spec_from_file_location("cleanup_duplicate_buckets", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bucket(bucket_id, content, **meta):
    return {
        "id": bucket_id,
        "content": content,
        "metadata": {
            "name": bucket_id,
            "type": "dynamic",
            "importance": 5,
            **meta,
        },
    }


class FakeBucketManager:
    def __init__(self):
        self.deleted = []

    async def delete(self, bucket_id):
        self.deleted.append(bucket_id)
        return True


class FakeEmbeddingEngine:
    def __init__(self):
        self.deleted = []

    def delete_embedding(self, bucket_id):
        self.deleted.append(bucket_id)


def test_exact_duplicate_plan_keeps_important_bucket_and_deletes_dynamic_copy():
    cleanup = _load_cleanup_module()
    buckets = [
        _bucket("keep", "小雨要把重复导入清理掉。", importance=9),
        _bucket("dupe", "小雨要把重复导入清理掉。", importance=5),
    ]

    plans = cleanup.exact_duplicate_plans(buckets, min_chars=5)

    assert len(plans) == 1
    assert plans[0].keep_id == "keep"
    assert plans[0].delete_ids == ["dupe"]


def test_exact_duplicate_plan_does_not_delete_pinned_or_permanent_bucket():
    cleanup = _load_cleanup_module()
    buckets = [
        _bucket("dynamic", "这条重要记忆被重复导入。", importance=5),
        _bucket("pinned", "这条重要记忆被重复导入。", pinned=True, importance=10),
        _bucket("permanent", "这条重要记忆被重复导入。", type="permanent", importance=8),
    ]

    plans = cleanup.exact_duplicate_plans(buckets, min_chars=5)

    assert len(plans) == 1
    assert plans[0].keep_id == "pinned"
    assert plans[0].delete_ids == ["dynamic"]


def test_exact_duplicate_plan_does_not_delete_bucket_with_comments():
    cleanup = _load_cleanup_module()
    buckets = [
        _bucket("plain", "这条记忆有一个带年轮的重复桶。", importance=5),
        _bucket("commented", "这条记忆有一个带年轮的重复桶。", comments=[{"content": "留下来的年轮"}], importance=4),
    ]

    plans = cleanup.exact_duplicate_plans(buckets, min_chars=5)

    assert plans == []


def test_exact_duplicate_plan_ignores_affect_anchor():
    cleanup = _load_cleanup_module()
    buckets = [
        _bucket("keep", "小雨决定重复清理只看正文主体。", importance=8),
        _bucket(
            "dupe",
            (
                "小雨决定重复清理只看正文主体。\n\n"
                "### affect_anchor\n\n"
                "> 小雨把旧信放到桌上。\n"
                "> Dbmaj9 -> Ab/C -> Bbm9\n\n"
                "含义：这只是温度。"
            ),
            importance=5,
        ),
    ]

    plans = cleanup.exact_duplicate_plans(buckets, min_chars=5)

    assert len(plans) == 1
    assert plans[0].keep_id == "keep"
    assert plans[0].delete_ids == ["dupe"]


def test_near_duplicate_pairs_are_reported_for_manual_review_only():
    cleanup = _load_cleanup_module()
    buckets = [
        _bucket("a", "小雨决定周末去杭州参加朋友婚礼，需要提前买高铁票并准备蓝色连衣裙。"),
        _bucket("b", "周末小雨要去杭州参加朋友的婚礼，她需要提前订高铁票，也想带上蓝色连衣裙。"),
        _bucket("c", "小雨下个月要去上海参加同事婚礼，需要订酒店和准备红包。"),
    ]

    pairs = cleanup.near_duplicate_pairs(buckets, threshold=88, min_chars=10)

    assert pairs[0][0:2] == ("a", "b")
    assert pairs[0][2] >= 88


def test_near_duplicate_pairs_can_exclude_exact_duplicate_pairs():
    cleanup = _load_cleanup_module()
    buckets = [
        _bucket("a", "第一句：完全重复。第二句：应该只出现在 exact。"),
        _bucket("b", "第一句：完全重复。第二句：应该只出现在 exact。"),
    ]

    pairs = cleanup.near_duplicate_pairs(buckets, threshold=88, min_chars=5, exclude_pairs={frozenset(("a", "b"))})

    assert pairs == []


def test_near_duplicate_pairs_can_compare_protected_bucket_with_safe_copy():
    cleanup = _load_cleanup_module()
    buckets = [
        _bucket("anchor", "小雨决定周末去杭州参加朋友婚礼，需要提前买高铁票。", pinned=True),
        _bucket("copy", "周末小雨要去杭州参加朋友婚礼，需要提前订高铁票。"),
    ]

    pairs = cleanup.near_duplicate_pairs(buckets, threshold=80, min_chars=10)

    assert pairs[0][0:2] == ("anchor", "copy")


def test_suggested_near_action_deletes_safe_copy_when_other_side_is_protected():
    cleanup = _load_cleanup_module()
    buckets = {
        "anchor": _bucket("anchor", "小雨决定周末去杭州参加朋友婚礼，需要提前买高铁票。", pinned=True),
        "copy": _bucket("copy", "周末小雨要去杭州参加朋友婚礼，需要提前订高铁票。"),
    }

    assert cleanup.suggested_near_action("anchor", "copy", buckets) == ("anchor", "copy")


def test_content_preview_keeps_first_two_sentences():
    cleanup = _load_cleanup_module()
    bucket = _bucket("a", "第一句。第二句！第三句不会展示。")

    assert cleanup.content_preview(bucket) == "第一句。第二句！"


def test_interactive_cleanup_deletes_exact_group_after_confirmation(monkeypatch):
    cleanup = _load_cleanup_module()
    bucket_mgr = FakeBucketManager()
    embedding_engine = FakeEmbeddingEngine()
    buckets = {
        "keep": _bucket("keep", "小雨把重复导入清理掉。", importance=9),
        "dupe": _bucket("dupe", "小雨把重复导入清理掉。", importance=5),
    }
    plan = cleanup.DuplicatePlan(
        key="same",
        keep_id="keep",
        delete_ids=["dupe"],
        bucket_ids=["keep", "dupe"],
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    deleted = asyncio.run(cleanup.interactive_cleanup(bucket_mgr, embedding_engine, [plan], [], buckets))

    assert deleted == ["dupe"]
    assert bucket_mgr.deleted == ["dupe"]
    assert embedding_engine.deleted == ["dupe"]


def test_interactive_cleanup_can_delete_suggested_near_duplicate_with_y(monkeypatch):
    cleanup = _load_cleanup_module()
    bucket_mgr = FakeBucketManager()
    embedding_engine = FakeEmbeddingEngine()
    buckets = {
        "a": _bucket("a", "小雨周末要去杭州参加朋友婚礼，提前订高铁票。", importance=8),
        "b": _bucket("b", "周末小雨去杭州参加朋友婚礼，需要提前买高铁票。", importance=5),
    }
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    deleted = asyncio.run(
        cleanup.interactive_cleanup(bucket_mgr, embedding_engine, [], [("a", "b", 82.0)], buckets)
    )

    assert deleted == ["b"]
    assert bucket_mgr.deleted == ["b"]
    assert embedding_engine.deleted == ["b"]


def test_interactive_cleanup_can_delete_near_duplicate_by_side_number(monkeypatch):
    cleanup = _load_cleanup_module()
    bucket_mgr = FakeBucketManager()
    embedding_engine = FakeEmbeddingEngine()
    buckets = {
        "a": _bucket("a", "小雨周末要去杭州参加朋友婚礼，提前订高铁票。", importance=8),
        "b": _bucket("b", "周末小雨去杭州参加朋友婚礼，需要提前买高铁票。", importance=5),
    }
    monkeypatch.setattr("builtins.input", lambda prompt: "2")

    deleted = asyncio.run(
        cleanup.interactive_cleanup(bucket_mgr, embedding_engine, [], [("a", "b", 82.0)], buckets)
    )

    assert deleted == ["b"]
    assert bucket_mgr.deleted == ["b"]
    assert embedding_engine.deleted == ["b"]
