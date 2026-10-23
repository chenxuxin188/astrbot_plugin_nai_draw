"""不依赖 AstrBot 的单元测试：python -m pytest tests"""

import asyncio
import io
import json
import sys
import zipfile
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audit  # noqa: E402
import nai_client  # noqa: E402
import prompting  # noqa: E402
from nai_client import ImageRequest, NovelAIClient, NovelAIError  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def make_zip(data: bytes = PNG) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("image_0.png", data)
    return buf.getvalue()


def make_request(**kw) -> ImageRequest:
    base = dict(
        model="nai-diffusion-4-5-full", prompt="1girl, smile", negative="lowres",
        width=832, height=1216, steps=28, scale=5.0, cfg_rescale=0.0,
        sampler="k_euler_ancestral", noise_schedule="karras", seed=123,
    )
    base.update(kw)
    return ImageRequest(**base)


# ---------------------------------------------------------------- nai_client


@pytest.mark.parametrize(
    ("text", "expected"),
    [("832x1216", (832, 1216)), (" 1216 X 832 ", (1216, 832)), ("1024×1024", (1024, 1024)),
     ("bad", (1, 2)), ("0x100", (1, 2)), (None, (1, 2))],
)
def test_parse_size(text, expected):
    assert nai_client.parse_size(text, (1, 2)) == expected


def test_fit_size_free_guard():
    assert nai_client.fit_size(832, 1216) == (832, 1216)
    assert nai_client.fit_size(1024, 1024) == (1024, 1024)
    w, h = nai_client.fit_size(1536, 2048)
    assert w * h <= nai_client.OPUS_FREE_MAX_PIXELS and w % 64 == 0 and h % 64 == 0
    assert nai_client.fit_size(1536, 2048, free_guard=False) == (1536, 2048)
    assert nai_client.fit_size(830, 1220, free_guard=False) == (768, 1216)
    assert nai_client.fit_size(10, 10) == (64, 64)


def test_is_opus_free():
    assert nai_client.is_opus_free("nai-diffusion-4-5-full", 832, 1216, 28)
    assert not nai_client.is_opus_free("nai-diffusion-4-5-full", 832, 1216, 29)
    assert not nai_client.is_opus_free("nai-diffusion-4-5-full", 1088, 1088, 28)
    assert not nai_client.is_opus_free("nai-diffusion-5-full", 832, 1216, 28)


def test_build_payload():
    body = nai_client.build_payload(make_request())
    p = body["parameters"]
    assert body["input"] == "1girl, smile" and body["action"] == "generate"
    assert body["model"] == "nai-diffusion-4-5-full"
    assert p["v4_prompt"]["caption"]["base_caption"] == "1girl, smile"
    assert p["v4_negative_prompt"]["caption"]["base_caption"] == "lowres"
    assert p["negative_prompt"] == "lowres"
    assert p["n_samples"] == 1 and p["seed"] == 123 and p["steps"] == 28
    assert p["deliberate_euler_ancestral_bug"] is False and p["prefer_brownian"] is True
    json.dumps(body)  # 必须可序列化

    other = nai_client.build_payload(make_request(sampler="k_euler", seed=0))["parameters"]
    assert "prefer_brownian" not in other
    assert 1 <= other["seed"] <= nai_client.MAX_SEED


def test_extract_image():
    assert nai_client.extract_image(make_zip()) == PNG
    assert nai_client.extract_image(PNG) == PNG
    with pytest.raises(NovelAIError):
        nai_client.extract_image(b"not an image")


def test_error_from_response():
    e = nai_client.error_from_response(401, {"statusCode": 401, "message": "Invalid token"})
    assert "Token" in str(e) and "Invalid token" in str(e) and not e.retryable
    locked = nai_client.error_from_response(429, {"message": "Concurrent generation is locked"})
    assert locked.retryable and locked.concurrent
    assert nai_client.error_from_response(409, {"message": "concurrent generation"}).concurrent
    assert not nai_client.error_from_response(409, {"message": "conflict"}).retryable
    assert not nai_client.error_from_response(402, {}).retryable
    # 可能已在服务端开始生成的错误不重试
    for status in (500, 504, 524):
        assert not nai_client.error_from_response(status, {}).retryable
    page = "<!DOCTYPE html><html><head><title>502 Bad Gateway</title></head><body>x</body></html>"
    bad_gateway = nai_client.error_from_response(502, page)
    assert bad_gateway.retryable and "<" not in str(bad_gateway) and "502 Bad Gateway" in str(bad_gateway)
    assert "<" not in str(nai_client.error_from_response(503, "<html><body>no title</body></html>"))


def test_subscription_helpers():
    sub = {"tier": 3, "active": True, "trainingStepsLeft": {"fixedTrainingStepsLeft": 10000,
                                                            "purchasedTrainingSteps": 500}}
    assert nai_client.anlas_total(sub) == 10500
    assert nai_client.anlas_total({}) is None
    assert nai_client.tier_name(3) == "Opus" and nai_client.tier_name(None) == "未知"
    assert nai_client.format_expiry(1767225600) == nai_client.format_expiry(1767225600000) != ""
    assert nai_client.format_expiry(None) == ""
    assert nai_client.mask_token("pst-abcdefghijklmnop") == "pst-…mnop"


def run(coro):
    return asyncio.run(coro)


def test_client_generate_success():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=make_zip())

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.generate(make_request(), token="pst-x", base_url="https://image.novelai.net/")
        finally:
            await client.close()

    assert run(go()) == PNG
    assert seen["url"] == "https://image.novelai.net/ai/generate-image"
    assert seen["auth"] == "Bearer pst-x"
    assert seen["body"]["parameters"]["width"] == 832


def test_client_generate_retries_on_429():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"statusCode": 429, "message": "Concurrent generation is locked"})
        return httpx.Response(200, content=make_zip())

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.generate(make_request(), token="t", retry_delay=0)
        finally:
            await client.close()

    assert run(go()) == PNG and len(calls) == 3


def run_generate(handler):
    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.generate(make_request(), token="t", retry_delay=0)
        finally:
            await client.close()

    return run(go())


def test_client_generate_retries_on_409_concurrent():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(409, json={"message": "Concurrent generation is locked"})
        return httpx.Response(200, content=make_zip())

    assert run_generate(handler) == PNG and len(calls) == 3


def test_client_generate_does_not_retry_timeout():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(NovelAIError) as info:
        run_generate(handler)
    assert len(calls) == 1 and "超时" in str(info.value)


def test_client_generate_retries_connect_error_once():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(NovelAIError):
        run_generate(handler)
    assert len(calls) == 2


def test_client_refuses_after_close():
    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=make_zip())))
        await client.close()
        await client.generate(make_request(), token="t", retry_delay=0)

    with pytest.raises(NovelAIError):
        run(go())


def test_normalize_urls():
    assert nai_client.normalize_base_url("") == nai_client.DEFAULT_BASE_URL
    assert nai_client.normalize_base_url(" image.novelai.net/ ") == "https://image.novelai.net"
    assert nai_client.normalize_base_url("http://relay.example.com/nai/") == "http://relay.example.com/nai"
    assert nai_client.normalize_proxy_url("  ") == ""
    assert nai_client.normalize_proxy_url("127.0.0.1:7890") == "http://127.0.0.1:7890"
    assert nai_client.normalize_proxy_url("socks5://127.0.0.1:7890") == "socks5://127.0.0.1:7890"
    assert nai_client.mask_url_credentials("http://user:pw@1.2.3.4:8080") == "http://***@1.2.3.4:8080"
    assert nai_client.mask_url_credentials("http://1.2.3.4:8080") == "http://1.2.3.4:8080"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [("image.novelai.net", "https://image.novelai.net/ai/generate-image"),
     ("https://relay.example.com/nai/", "https://relay.example.com/nai/ai/generate-image")],
)
def test_client_uses_configured_base_url(base_url, expected):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=make_zip())

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.generate(make_request(), token="t", base_url=base_url)
        finally:
            await client.close()

    assert run(go()) == PNG and seen == [expected]


def test_client_invalid_proxy_reports_error():
    async def go():
        client = NovelAIClient()
        try:
            await client.generate(make_request(), token="t", proxy="ftp://127.0.0.1:21")
        finally:
            await client.close()

    with pytest.raises(NovelAIError) as info:
        run(go())
    assert "代理地址无效" in str(info.value) and not info.value.retryable


def test_client_proxy_error_retried_once():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ProxyError("407 Proxy Authentication Required", request=request)

    with pytest.raises(NovelAIError) as info:
        run_generate(handler)
    assert len(calls) == 2 and "代理" in str(info.value)


def test_client_unsupported_protocol_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.UnsupportedProtocol("Request URL has an unsupported protocol 'ftp://'.", request=request)

    with pytest.raises(NovelAIError) as info:
        run_generate(handler)
    assert len(calls) == 1 and "协议" in str(info.value)


def test_client_generate_no_retry_on_401():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"statusCode": 401, "message": "Unauthorized"})

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            await client.generate(make_request(), token="t", retry_delay=0)
        finally:
            await client.close()

    with pytest.raises(NovelAIError) as info:
        run(go())
    assert info.value.status == 401 and len(calls) == 1


def test_client_subscription():
    def handler(request):
        assert request.url.path == "/user/subscription"
        return httpx.Response(200, json={"tier": 3, "active": True})

    async def go():
        client = NovelAIClient(transport=httpx.MockTransport(handler))
        try:
            return await client.subscription(token="t")
        finally:
            await client.close()

    assert run(go())["tier"] == 3


# ---------------------------------------------------------------- prompting


def test_normalize_orientation():
    assert prompting.normalize_orientation("Portrait") == "portrait"
    assert prompting.normalize_orientation("横图") == "landscape"
    assert prompting.normalize_orientation("square") == "square"
    assert prompting.normalize_orientation("diagonal") == ""
    assert prompting.normalize_orientation(None) == ""


def test_parse_tag_output_json():
    raw = '<think>hmm</think>```json\n{"prompt": "1girl, cat ears, 1girl", "negative": "", "orientation": "landscape"}\n```'
    out = prompting.parse_tag_output(raw)
    assert out == {"prompt": "1girl, cat ears", "negative": "", "orientation": "landscape"}


def test_parse_tag_output_plain_text():
    out = prompting.parse_tag_output("Prompt: 1boy，sword\nnight sky")
    assert out["prompt"] == "1boy, sword, night sky" and out["orientation"] == ""


def test_parse_tag_output_list_and_bad_orientation():
    out = prompting.parse_tag_output('{"prompt": ["a", "b"], "orientation": "weird"}')
    assert out["prompt"] == "a, b" and out["orientation"] == ""


def test_contains_non_english():
    assert prompting.contains_non_english("1girl, 猫耳")
    assert prompting.contains_non_english("ねこ") and prompting.contains_non_english("고양이")
    assert prompting.contains_non_english("ｶﾀｶﾅ")  # 半角片假名
    assert prompting.contains_non_english("кошка") and prompting.contains_non_english("แมว")
    assert prompting.contains_non_english("1girl。")  # 中文标点
    assert not prompting.contains_non_english("1girl, cat ears, pokémon, 1.2::smile::, {{blush}}, ♥, :3")
    assert not prompting.contains_non_english("ｃａｔ　ｅａｒｓ，１ｇｉｒｌ")  # 全角字母和标点会转成半角
    assert not prompting.contains_non_english("")


def test_drop_non_english_tags():
    kept, removed = prompting.drop_non_english_tags("1girl, 白色长发, cat ears，smile、穿着JK制服")
    assert kept == "1girl, cat ears, smile" and removed == ["白色长发", "穿着JK制服"]
    long_text = ", ".join(f"tag{i}" for i in range(500))
    assert prompting.drop_non_english_tags(long_text)[0] == long_text  # 不截断
    assert prompting.drop_non_english_tags("") == ("", [])


def test_drop_non_english_tags_config_cases():
    # 每行一个标签的配置：只去掉中文那一行
    kept, removed = prompting.drop_non_english_tags("nsfw\nlowres\n低质量\nbad hands")
    assert kept == "nsfw, lowres, bad hands" and removed == ["低质量"]
    # 不去重，跨逗号的权重分组保持原样
    weights = "1.2::cat ears, smile::, 0.8::dog ears, smile::, very aesthetic"
    assert prompting.drop_non_english_tags(weights)[0] == weights
    # 去掉分组中的中文标签时保留未闭合的权重 / 括号符号
    assert prompting.drop_non_english_tags("1.2::猫耳, smile::, blush")[0] == "1.2::, smile::, blush"
    assert prompting.drop_non_english_tags("{{白发, smile}}")[0] == "{{, smile}}"
    assert prompting.drop_non_english_tags("1.2::猫耳::, smile")[0] == "smile"
    # 、。 视为分隔符，「」【】 等括号去掉
    assert prompting.drop_non_english_tags("smile。blush、「cat ears」")[0] == "smile, blush, cat ears"


def test_choose_tag_result():
    first = {"prompt": "1girl, 白色长发, cat ears, smile", "negative": "blurry", "orientation": "portrait"}
    better = prompting.choose_tag_result(first, {"prompt": "1girl, white hair, cat ears", "negative": "", "orientation": ""})
    assert better == {"prompt": "1girl, white hair, cat ears", "negative": "blurry", "orientation": "portrait"}
    # 改写结果更差（英文标签更少或全是中文）时沿用第一次的结果
    assert prompting.choose_tag_result(first, {"prompt": "一个女孩, 猫耳", "negative": "", "orientation": ""}) is first
    assert prompting.choose_tag_result(first, {"prompt": "", "negative": "", "orientation": ""}) is first
    assert prompting.english_tag_count("1girl, 白发, smile") == 2 and prompting.english_tag_count("") == 0


def test_tag_system_prompt_is_configurable():
    custom = prompting.build_tag_system_prompt("自定义：只画风景。", sfw=False)
    assert custom.startswith("自定义：只画风景。")
    # 输出格式、英文要求和安全规则始终附加
    assert prompting.TAG_OUTPUT_RULES in custom and "ENGLISH ONLY" in custom and "minors" in custom
    default = prompting.build_tag_system_prompt("   ")
    assert default.startswith(prompting.DEFAULT_TAG_PROMPT)
    assert "english prompt that will be sent directly to novelai" in prompting.DEFAULT_TAG_PROMPT.lower()
    schema = json.loads((Path(__file__).resolve().parent.parent / "_conf_schema.json").read_text(encoding="utf-8"))
    assert schema["prompt"]["items"]["tag_system_prompt"]["default"] == prompting.DEFAULT_TAG_PROMPT


def test_parse_tag_output_normalizes_fullwidth():
    out = prompting.parse_tag_output('{"prompt": "１girl，ｃａｔ ears", "negative": "", "orientation": "portrait"}')
    assert out["prompt"] == "1girl, cat ears"


def test_tag_prompts_require_english():
    assert "ENGLISH ONLY" in prompting.build_tag_system_prompt()
    assert "minors" in prompting.build_tag_system_prompt(sfw=False)
    retry = prompting.build_tag_retry_prompt("画一只猫", "square", '{"prompt": "猫"}')
    assert "画一只猫" in retry and '{"prompt": "猫"}' in retry and "English" in retry


def test_clean_tags_limit():
    tags = prompting.clean_tags(", ".join(f"tag{i}" for i in range(1000)), limit=50)
    assert len(tags) <= 50 and not tags.endswith(",")


def test_join_tags():
    assert prompting.join_tags("a, b,", "", " c ") == "a, b, c"
    assert prompting.join_tags("", None) == ""


def test_strip_think():
    assert prompting.strip_think("<think>x</think>hello") == "hello"
    assert prompting.strip_think("reasoning...</think>\nhi") == "hi"
    assert prompting.strip_think("plain") == "plain"


def test_clean_line():
    assert prompting.clean_line('“我开始画啦！”\n', 40) == "我开始画啦！"
    assert prompting.clean_line("第一行\n第二行", 40) == "第一行 第二行"
    assert len(prompting.clean_line("长" * 500, 20)) == 41
    # 超长时尽量截在句末标点
    assert prompting.clean_line("好" * 15 + "！" + "多" * 60, 20) == "好" * 15 + "！"


def test_prompt_builders():
    sp = prompting.build_tag_system_prompt(sfw=True, extra="watercolor")
    assert "safe for work" in sp and "watercolor" in sp
    assert "safe for work" not in prompting.build_tag_system_prompt(sfw=False)
    assert "Requested orientation: square" in prompting.build_tag_user_prompt("猫", "square")
    assert "猫娘" in prompting.build_persona_system_prompt("你是猫娘", 30)
    for kind in ("start", "success", "fail"):
        text = prompting.build_persona_user_prompt(kind, description="一只猫", prompt="cat", error="boom")
        assert "一只猫" in text and "{" not in text
        assert prompting.FALLBACK_LINES[kind]
    assert "正在画" in prompting.build_persona_user_prompt("start", description="一只猫")
    assert "正在画" in prompting.FALLBACK_LINES["start"]


def test_persona_prompt_templates():
    custom = prompting.build_persona_user_prompt(
        "success", "画了{description}，标签{prompt}，保留{未知}和{{花括号}}", description="猫", prompt="cat")
    assert custom == "画了猫，标签cat，保留{未知}和{{花括号}}"
    assert prompting.build_persona_user_prompt("fail", "  ", error="x") == \
        prompting.build_persona_user_prompt("fail", error="x")  # 空模板使用默认
    assert len(prompting.build_persona_user_prompt("fail", "{error}", error="e" * 1000)) == 200


def test_schema_persona_prompt_defaults_match_code():
    schema = json.loads((Path(__file__).resolve().parent.parent / "_conf_schema.json").read_text(encoding="utf-8"))
    items = schema["persona"]["items"]
    for kind, template in prompting.DEFAULT_PERSONA_PROMPTS.items():
        assert items[f"{kind}_prompt"]["default"] == template


# ---------------------------------------------------------------- audit


def test_audit_page_and_trim(tmp_path):
    log = audit.AuditLog(tmp_path / "audit.jsonl", max_records=10)

    async def go():
        for i in range(30):
            await log.append({"status": "success", "request": f"r{i}"})
        return await log.page(1, 5), await log.page(99, 5)

    (first, page, pages, total), (_, last_page, _, _) = run(go())
    assert total <= 10 + 10 and total >= 10
    assert page == 1 and pages == -(-total // 5) and last_page == pages
    assert first[0]["request"] == "r29"  # 从新到旧


def test_audit_append_survives_cancellation(tmp_path):
    log = audit.AuditLog(tmp_path / "audit.jsonl")

    async def go():
        tasks = [asyncio.create_task(log.append({"status": "success", "request": str(i)})) for i in range(50)]
        await asyncio.sleep(0)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return await log.page(1, 100)

    _, _, _, total = run(go())
    assert total == 50


def test_audit_skips_corrupt_lines(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text('{"status":"success"}\nnot json\n[1]\n', encoding="utf-8")
    records, _, _, total = run(audit.AuditLog(path).page(1, 10))
    assert total == 1 and records[0]["_no"] == 1


def test_format_record():
    ok = audit.format_record({
        "_no": 3, "status": "success", "time": "2026-10-23T14:05:11+08:00", "user_name": "小明",
        "user_id": "123", "group_id": "456", "model": "nai-diffusion-4-5-full", "size": "832x1216",
        "steps": 28, "seed": 9, "duration": 12.3, "anlas_cost": 0, "request": "画猫", "prompt": "cat",
    })
    assert "#3 ✅ 2026-10-23 14:05:11" in ok and "群 456" in ok and "消耗 0 Anlas" in ok
    bad = audit.format_record({"status": "failed", "stage": "generate", "error": "Token 无效", "request": "x"})
    assert "❌" in bad and "失败（生成图片）：Token 无效" in bad and "私聊" in bad
