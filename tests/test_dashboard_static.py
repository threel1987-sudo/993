from pathlib import Path


def test_dashboard_comments_show_author_and_time_without_emotion_fields():
    html = Path("dashboard.html").read_text(encoding="utf-8")

    assert "var commentTime = c.original_feel_created || c.created || '';" in html
    assert "let dashboardAiAuthor = 'Haven';" in html
    assert "function commentAuthorName(comment)" in html
    assert '<div class="comment-meta">' in html
    assert '<span class="comment-author">' in html
    assert '<span class="comment-time">' in html

    comments_block = html.split("var commentsHtml = comments.length", 1)[1].split("var commentFormHtml", 1)[0]
    assert "commentAuthorName(c)" in comments_block
    assert "c.valence" not in comments_block
    assert "c.arousal" not in comments_block


def test_dashboard_comment_enter_submit_has_no_visible_send_key():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    form_block = html.split("var commentFormHtml =", 1)[1].split("content.innerHTML =", 1)[0]

    assert "handleCommentKeydown(event)" in form_block
    assert "comment-send-button" not in html
    assert 'aria-label="发送"' not in form_block


def test_dashboard_exposes_gateway_memory_cooldown_settings():
    html = Path("dashboard.html").read_text(encoding="utf-8")

    assert "<h3>记忆浮现</h3>" in html
    assert 'id="cfg-gateway-cooldown"' in html
    assert 'id="cfg-gateway-rounds"' in html
    assert "cfg.gateway.cooldown_hours" in html
    assert "cfg.gateway.skip_recent_rounds" in html
    assert "cooldown_hours: floatValue('cfg-gateway-cooldown', 6)" in html
    assert "skip_recent_rounds: numberValue('cfg-gateway-rounds', 5)" in html


def test_dashboard_exposes_reflection_affect_anchor_switches():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]

    assert "<h3>记忆关系整理</h3>" in html
    assert 'id="cfg-reflection-memory-anchor"' in html
    assert 'id="cfg-reflection-weather-anchor"' in html
    assert "cfg.reflection.memory_affect_anchor_enabled" in load_block
    assert "cfg.reflection.relationship_weather_affect_anchor_enabled" in load_block
    assert "memory_affect_anchor_enabled: document.getElementById('cfg-reflection-memory-anchor').value === 'true'," in save_block
    assert "relationship_weather_affect_anchor_enabled: document.getElementById('cfg-reflection-weather-anchor').value === 'true'," in save_block


def test_dashboard_dream_background_control_uses_auto_enabled_only():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]
    dream_block = save_block.split("dream: {", 1)[1].split("gateway: {", 1)[0]
    dream_lines = [line.strip() for line in dream_block.splitlines()]

    assert "document.getElementById('cfg-dream-enabled').value = cfg.dream.auto_enabled ? 'true' : 'false';" in load_block
    assert "document.getElementById('cfg-dream-enabled').value = cfg.dream.enabled" not in load_block
    assert "auto_enabled: document.getElementById('cfg-dream-enabled').value === 'true'," in dream_lines
    assert "enabled: document.getElementById('cfg-dream-enabled').value === 'true'," not in dream_lines
    assert "surface_enabled: document.getElementById('cfg-dream-surface').value === 'true'," in dream_lines


def test_dashboard_config_number_zero_values_are_preserved():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    load_block = html.split("async function loadConfig()", 1)[1].split("async function saveConfig", 1)[0]
    save_block = html.split("async function saveConfig", 1)[1].split("var keyVal =", 1)[0]

    assert "document.getElementById('cfg-dehy-temp').value = cfg.dehydration.temperature ?? 0.1;" in load_block
    assert "document.getElementById('cfg-merge').value = cfg.merge_threshold ?? 75;" in load_block
    assert "temperature: floatValue('cfg-dehy-temp', 0.1)," in save_block
    assert "merge_threshold: numberValue('cfg-merge', 75)," in save_block
    assert "cfg.dehydration.temperature || 0.1" not in load_block
    assert "cfg.merge_threshold || 75" not in load_block
    assert "parseFloat(document.getElementById('cfg-dehy-temp').value) || 0.1" not in save_block
    assert "parseInt(document.getElementById('cfg-merge').value) || 75" not in save_block


def test_dashboard_import_file_input_resets_after_selection():
    html = Path("dashboard.html").read_text(encoding="utf-8")
    import_block = html.split("// --- Import functions ---", 1)[1].split("async function pollImportStatus", 1)[0]

    assert "const selectedFile = fileInput.files[0];" in import_block
    assert "fileInput.value = '';" in import_block
    assert "startImport(selectedFile);" in import_block
