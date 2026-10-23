"""LLM 提示词构造与输出解析（不依赖 AstrBot，便于单独测试）。"""

import json
import re

ORIENTATIONS = ("portrait", "landscape", "square")

_ORIENTATION_ALIASES = {
    "竖": "portrait", "竖图": "portrait", "竖屏": "portrait", "纵向": "portrait",
    "vertical": "portrait", "tall": "portrait",
    "横": "landscape", "横图": "landscape", "横屏": "landscape", "横向": "landscape",
    "horizontal": "landscape", "wide": "landscape",
    "方": "square", "方图": "square", "正方形": "square",
}

MAX_PROMPT_CHARS = 2000

FALLBACK_LINES = {
    "start": "收到～这就开始画，稍等一下哦！",
    "success": "画好啦！看看喜不喜欢～",
    "fail": "呜……这次没能画出来，好失落。",
}

_THINK_RE = re.compile(r"<(think|thinking)>.*?</\1>", re.S | re.I)

TAG_SYSTEM_PROMPT = """You are an expert prompt writer for NovelAI Diffusion, an anime-style image model that understands Danbooru tags.
Convert the user's drawing request (it may be in any language) into a NovelAI prompt.

Output ONLY one JSON object, without markdown code fences or any other text:
{"prompt": "<tags>", "negative": "<tags or empty string>", "orientation": "portrait|landscape|square"}

Rules for "prompt":
- English, lowercase, comma-separated Danbooru-style tags; short natural-language phrases are allowed only for things tags cannot express.
- Order: subject count (1girl, 2girls, 1boy, no humans, ...); for known characters the exact Danbooru character tag (keep a parenthesised qualifier only when the Danbooru tag itself has one, e.g. furina (genshin impact); otherwise none, e.g. hatsune miku) followed by the series/copyright as a separate tag (e.g. hatsune miku, vocaloid); then appearance (hair, eyes, body), clothing, expression, pose/action, held objects, background/scene, lighting, composition/camera angle, art style.
- 15-40 tags. Be concrete and visual; turn abstract ideas into visible elements. Fill in reasonable details the user did not specify.
- Do NOT add quality or aesthetic tags (masterpiece, best quality, very aesthetic, absurdres, ...); they are appended automatically.
- Do not add artist tags unless the user explicitly asks for an artist's style; then write each one as artist:<danbooru artist tag> (e.g. artist:wlop).

Rules for "negative": only extra things to avoid that are specific to this request (may be an empty string). Do not repeat generic quality negatives.

Rules for "orientation": portrait for single characters / full body; landscape for scenery, wide scenes or several characters side by side; square for icons, avatars or close-up faces."""

TAG_SFW_RULE = (
    "Keep the image safe for work: no nudity or sexual content. "
    "If the request is explicit, draw a tasteful safe-for-work version instead."
)


def normalize_orientation(value: object) -> str:
    """把各种写法的方向统一为 portrait / landscape / square，无法识别时返回空串。"""
    text = str(value or "").strip().lower()
    if text in ORIENTATIONS:
        return text
    return _ORIENTATION_ALIASES.get(text, "")


def strip_think(text: str) -> str:
    """去掉推理模型输出的 <think> 段落。"""
    text = _THINK_RE.sub("", text or "")
    # 只有结束标签（开头被截掉）时，取最后一个结束标签之后的内容
    return re.split(r"</think(?:ing)?>", text, flags=re.I)[-1].strip()


def join_tags(*parts: str) -> str:
    """用 ", " 拼接多段 tag，忽略空段。"""
    out = []
    for part in parts:
        part = re.sub(r"\s+", " ", str(part or "")).strip(" ,，")
        if part:
            out.append(part)
    return ", ".join(out)


def clean_tags(text: object, limit: int = MAX_PROMPT_CHARS) -> str:
    """规范化 tag 串：统一逗号、去重、去空白，并在 tag 边界处截断到 limit 字符。"""
    if isinstance(text, (list, tuple)):
        text = ", ".join(str(t) for t in text)
    text = str(text or "").replace("，", ",").replace("\n", ",")
    seen: set[str] = set()
    out: list[str] = []
    length = 0
    for tag in text.split(","):
        tag = re.sub(r"\s+", " ", tag).strip()
        if not tag or tag.lower() in seen:
            continue
        if length + len(tag) + 2 > limit:
            break
        seen.add(tag.lower())
        out.append(tag)
        length += len(tag) + 2
    return ", ".join(out)


def build_tag_system_prompt(*, sfw: bool = True, extra: str = "") -> str:
    prompt = TAG_SYSTEM_PROMPT
    if sfw:
        prompt += "\n\nSafety: " + TAG_SFW_RULE
    extra = (extra or "").strip()
    if extra:
        prompt += "\n\nAdditional requirements from the bot admin:\n" + extra
    return prompt


def build_tag_user_prompt(description: str, orientation: str = "") -> str:
    text = f"Drawing request:\n{description.strip()}"
    if orientation:
        text += f"\n\nRequested orientation: {orientation}"
    return text


def parse_tag_output(text: str) -> dict:
    """解析 LLM 返回的 JSON；不是 JSON 时把整段文本当作正面 prompt。"""
    text = strip_think(text)
    text = re.sub(r"```[a-zA-Z]*", "", text).strip()
    data = None
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
    if isinstance(data, dict):
        prompt = data.get("prompt") or data.get("positive") or data.get("tags") or ""
        negative = data.get("negative") or data.get("negative_prompt") or ""
        orientation = data.get("orientation") or ""
    else:
        prompt = re.sub(r"^\s*(positive\s*)?prompt\s*[:：]\s*", "", text, flags=re.I)
        negative, orientation = "", ""
    return {
        "prompt": clean_tags(prompt),
        "negative": clean_tags(negative, limit=MAX_PROMPT_CHARS // 2),
        "orientation": normalize_orientation(orientation),
    }


def build_persona_system_prompt(persona: str, max_len: int) -> str:
    base = (persona or "").strip() or "你是一个友好、可爱的聊天伙伴。"
    return (
        f"{base}\n\n"
        "【当前情景】你正在聊天中亲自为对方画画。\n"
        f"【输出要求】保持上面设定的人格、语气和称呼习惯，只输出一句不超过 {max_len} 字的口语化短句；"
        "不要解释、不要加引号、不要使用 Markdown，不要提及提示词、模型、AI 等幕后细节。"
    )


def build_persona_user_prompt(kind: str, *, description: str = "", prompt: str = "", error: str = "") -> str:
    description = description.strip()[:300]
    if kind == "start":
        return f"对方想让你画：{description}\n你现在准备动笔，说一句开工前的话。"
    if kind == "success":
        return (
            f"对方想让你画：{description}\n"
            f"你已经画好并把图发给了对方，画面内容（英文标签）：{prompt[:400]}\n"
            "用一句话简短评价一下这幅作品。"
        )
    return (
        f"对方想让你画：{description}\n"
        f"但是这次画画失败了（原因：{error[:200]}）。\n"
        "说一句失落、遗憾的话，不用复述错误细节。"
    )


def clean_line(text: str, max_len: int) -> str:
    """整理人格台词：去掉思考段落、合并多行、去掉包裹引号，并限制长度。

    LLM 对字数的把握不精确，超过 max_len 的 2 倍才截断，并尽量截在句末标点处。
    """
    text = strip_think(text)
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    text = text.strip().strip("\"'“”‘’「」『』").strip()
    max_len = max(10, max_len)
    limit = max_len * 2
    if len(text) > limit:
        cut = text[:limit]
        end = max(cut.rfind(c) for c in "。！？!?~～…")
        text = cut[:end + 1] if end >= max_len // 2 else cut.rstrip() + "…"
    return text
