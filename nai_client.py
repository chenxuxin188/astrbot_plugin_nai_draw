"""NovelAI 官方 API 的精简客户端（生成图片 / 查询订阅与 Anlas）。

本模块只依赖标准库与 httpx，不依赖 AstrBot，便于单独测试。
"""

import asyncio
import html
import io
import math
import random
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime

import httpx

DEFAULT_BASE_URL = "https://image.novelai.net"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
)

# Opus 会员免费出图条件：单张、无底图、总像素 ≤ 1024x1024、步数 ≤ 28（V5 模型除外）
OPUS_FREE_MAX_PIXELS = 1024 * 1024
OPUS_FREE_MAX_STEPS = 28
MAX_SEED = 4294967295 - 7

TIER_NAMES = {0: "Paper", 1: "Tablet", 2: "Scroll", 3: "Opus"}


class NovelAIError(Exception):
    """NovelAI 请求失败。message 为可直接展示给用户的中文说明。

    retryable 只在请求确定没有被处理时为 True（连接失败、并发锁定、网关暂时不可用），
    避免超时后重复提交导致重复生成、重复扣费。
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False,
                 concurrent: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.concurrent = concurrent


@dataclass
class ImageRequest:
    model: str
    prompt: str
    negative: str
    width: int
    height: int
    steps: int
    scale: float
    cfg_rescale: float
    sampler: str
    noise_schedule: str
    seed: int = 0


def parse_size(text: str, default: tuple[int, int] = (832, 1216)) -> tuple[int, int]:
    """解析 "832x1216" 形式的尺寸，失败时返回 default。"""
    try:
        w, h = str(text).lower().replace("×", "x").replace("*", "x").split("x", 1)
        width, height = int(w.strip()), int(h.strip())
    except (ValueError, AttributeError):
        return default
    if width <= 0 or height <= 0:
        return default
    return width, height


def fit_size(width: int, height: int, *, free_guard: bool = True) -> tuple[int, int]:
    """把尺寸对齐到 64 的倍数；开启 free_guard 时按比例缩小到 Opus 免费像素上限内。"""
    width, height = max(64, int(width)), max(64, int(height))
    if free_guard and width * height > OPUS_FREE_MAX_PIXELS:
        ratio = math.sqrt(OPUS_FREE_MAX_PIXELS / (width * height))
        width, height = width * ratio, height * ratio
    return max(64, int(width) // 64 * 64), max(64, int(height) // 64 * 64)


def is_v5_model(model: str) -> bool:
    return str(model).startswith("nai-diffusion-5")


def is_opus_free(model: str, width: int, height: int, steps: int) -> bool:
    """参数是否满足 Opus 免费出图条件（不含订阅状态判断）。"""
    return (
        not is_v5_model(model)
        and width * height <= OPUS_FREE_MAX_PIXELS
        and steps <= OPUS_FREE_MAX_STEPS
    )


def random_seed() -> int:
    return random.randint(1, MAX_SEED)


def build_payload(req: ImageRequest) -> dict:
    """构造 /ai/generate-image 的请求体（V3 / V4 / V4.5 通用格式）。"""
    seed = req.seed if 0 < req.seed <= MAX_SEED else random_seed()
    params = {
        "params_version": 3,
        "width": req.width,
        "height": req.height,
        "scale": float(req.scale),
        "sampler": req.sampler,
        "steps": int(req.steps),
        "n_samples": 1,
        "seed": seed,
        "noise_schedule": req.noise_schedule,
        "cfg_rescale": float(req.cfg_rescale),
        "ucPreset": 0,
        "qualityToggle": False,
        "sm": False,
        "sm_dyn": False,
        "dynamic_thresholding": False,
        "controlnet_strength": 1,
        "legacy": False,
        "legacy_v3_extend": False,
        "legacy_uc": False,
        "add_original_image": True,
        "skip_cfg_above_sigma": None,
        "use_coords": False,
        "normalize_reference_strength_multiple": True,
        "characterPrompts": [],
        "negative_prompt": req.negative,
        "v4_prompt": {
            "caption": {"base_caption": req.prompt, "char_captions": []},
            "use_coords": False,
            "use_order": True,
        },
        "v4_negative_prompt": {
            "caption": {"base_caption": req.negative, "char_captions": []},
            "legacy_uc": False,
        },
        "reference_image_multiple": [],
        "reference_information_extracted_multiple": [],
        "reference_strength_multiple": [],
    }
    if req.sampler == "k_euler_ancestral":
        params["deliberate_euler_ancestral_bug"] = False
        params["prefer_brownian"] = True
    return {"input": req.prompt, "model": req.model, "action": "generate", "parameters": params}


def extract_image(body: bytes) -> bytes:
    """从生成接口的响应中取出图片（通常是包含 image_0.png 的 ZIP）。"""
    if body[:8] == b"\x89PNG\r\n\x1a\n" or body[:3] == b"\xff\xd8\xff" or body[:4] == b"RIFF":
        return body
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                raise NovelAIError("NovelAI 返回了空的压缩包")
            return zf.read(names[0])
    except zipfile.BadZipFile as e:
        raise NovelAIError("NovelAI 返回的数据无法解析为图片") from e


_STATUS_TEXT = {
    400: "请求参数有误",
    401: "Token 无效或已过期",
    402: "订阅无效或 Anlas 不足",
    403: "请求被拒绝（可能是网络或地区限制）",
    404: "接口地址不存在，请检查 base_url",
    409: "账户正在进行其它生成",
    429: "账户正在进行其它生成或请求过于频繁",
}


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)


def error_from_response(status: int, body: object) -> NovelAIError:
    """根据 HTTP 状态码与响应体构造 NovelAIError。"""
    detail = ""
    if isinstance(body, dict):
        detail = str(body.get("message") or body.get("error") or "")
    elif body:
        detail = str(body)
        # Cloudflare 等返回的 HTML 错误页只保留标题
        if detail.lstrip().startswith("<") or "<html" in detail[:1000].lower():
            match = _TITLE_RE.search(detail)
            detail = html.unescape(match.group(1)) if match else ""
    detail = " ".join(detail.split())[:200]
    if status >= 500:
        text = "NovelAI 服务器错误"
    else:
        text = _STATUS_TEXT.get(status, "请求失败")
    concurrent = status == 429 or (status == 409 and "concurrent" in detail.lower())
    # 500/504/524 等可能已在服务端开始生成，不自动重试
    retryable = concurrent or status in (502, 503)
    message = f"{text}（HTTP {status}）"
    if detail:
        message += f"：{detail}"
    return NovelAIError(message, status=status, retryable=retryable, concurrent=concurrent)


def anlas_total(subscription: dict) -> int | None:
    """订阅 Anlas + 购买 Anlas；取不到时返回 None。"""
    steps = subscription.get("trainingStepsLeft") if isinstance(subscription, dict) else None
    if not isinstance(steps, dict):
        return None
    fixed, bought = steps.get("fixedTrainingStepsLeft"), steps.get("purchasedTrainingSteps")
    if isinstance(fixed, int) and isinstance(bought, int):
        return fixed + bought
    return None


def tier_name(tier: object) -> str:
    return TIER_NAMES.get(tier, str(tier)) if isinstance(tier, int) else "未知"


def format_expiry(value: object) -> str:
    """把 expiresAt（秒或毫秒时间戳）格式化为日期，无法解析时返回空串。"""
    if not isinstance(value, (int, float)) or value <= 0:
        return ""
    if value > 1e12:
        value /= 1000
    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def mask_token(token: str) -> str:
    token = str(token or "")
    if len(token) <= 10:
        return "*" * len(token)
    return f"{token[:4]}…{token[-4:]}"


class NovelAIClient:
    """复用连接的异步客户端；连接参数变化时自动重建。"""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None):
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._key: tuple | None = None
        # 配置变更后旧客户端可能仍有请求在进行，统一在 close() 时关闭
        self._stale: list[httpx.AsyncClient] = []
        self._closed = False
        self._lock = asyncio.Lock()

    async def _http(self, base_url: str, proxy: str) -> httpx.AsyncClient:
        key = ((base_url or DEFAULT_BASE_URL).strip().rstrip("/"), (proxy or "").strip() or None)
        async with self._lock:
            if self._closed:
                raise NovelAIError("插件已停用或正在重载，请求已取消")
            if self._client is None or self._key != key:
                if self._client is not None:
                    self._stale.append(self._client)
                self._client = httpx.AsyncClient(
                    base_url=key[0],
                    proxy=key[1],
                    headers={
                        "User-Agent": USER_AGENT,
                        "Origin": "https://novelai.net",
                        "Referer": "https://novelai.net/",
                    },
                    transport=self._transport,
                )
                self._key = key
            return self._client

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            clients = self._stale + ([self._client] if self._client is not None else [])
            self._client, self._key, self._stale = None, None, []
        for client in clients:
            await client.aclose()

    async def _request(self, method: str, path: str, *, token: str, base_url: str, proxy: str,
                       timeout: float, **kwargs) -> httpx.Response:
        client = await self._http(base_url, proxy)
        headers = {"Authorization": f"Bearer {token}", **kwargs.pop("headers", {})}
        try:
            resp = await client.request(
                method, path, headers=headers, timeout=httpx.Timeout(timeout, connect=15), **kwargs
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
            # 请求还没有发出，可以安全重试
            raise NovelAIError(f"无法连接 NovelAI（{type(e).__name__}: {e}）", retryable=True) from e
        except httpx.TimeoutException as e:
            raise NovelAIError(f"请求 NovelAI 超时（{type(e).__name__}）", retryable=False) from e
        except httpx.HTTPError as e:
            raise NovelAIError(f"与 NovelAI 的连接中断（{type(e).__name__}: {e}）", retryable=False) from e
        if not 200 <= resp.status_code < 300:
            try:
                body = resp.json()
            except ValueError:
                body = resp.text
            raise error_from_response(resp.status_code, body)
        return resp

    async def generate(self, req: ImageRequest, *, token: str, base_url: str = DEFAULT_BASE_URL,
                       proxy: str = "", timeout: float = 120, retry_delay: float = 5) -> bytes:
        """生成一张图片并返回图片字节。

        并发锁定（429 / 409 concurrent）最多重试 2 次；连接失败、502/503 重试 1 次。
        超时等无法确定服务端是否已开始生成的错误不重试。
        """
        payload = build_payload(req)
        lock_retries = other_retries = 0
        while True:
            try:
                resp = await self._request(
                    "POST", "/ai/generate-image", token=token, base_url=base_url, proxy=proxy,
                    timeout=timeout, json=payload,
                    headers={"Content-Type": "application/json", "Accept": "application/zip"},
                )
                return await asyncio.to_thread(extract_image, resp.content)
            except NovelAIError as e:
                if not e.retryable:
                    raise
                if e.concurrent:
                    lock_retries += 1
                    if lock_retries > 2:
                        raise
                else:
                    other_retries += 1
                    if other_retries > 1:
                        raise
                await asyncio.sleep(retry_delay)

    async def subscription(self, *, token: str, base_url: str = DEFAULT_BASE_URL, proxy: str = "",
                           timeout: float = 20) -> dict:
        """查询订阅信息（含 Anlas 余额）。"""
        resp = await self._request(
            "GET", "/user/subscription", token=token, base_url=base_url, proxy=proxy,
            timeout=timeout, headers={"Accept": "application/json"},
        )
        try:
            data = resp.json()
        except ValueError as e:
            raise NovelAIError("订阅信息不是有效的 JSON") from e
        if not isinstance(data, dict):
            raise NovelAIError("订阅信息格式异常")
        return data
