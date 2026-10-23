"""LLM 提示词构造与输出解析（不依赖 AstrBot，便于单独测试）。"""

import json
import re
import unicodedata

ORIENTATIONS = ("portrait", "landscape", "square")

# 中日韩文字、假名、谚文、注音及全角符号：NovelAI 只理解英文标签，这些字符不能出现在最终 prompt 中
_NON_ENGLISH_RE = re.compile(
    "[⺀-⿟　-〿぀-ヿ㄀-ㇿ㈀-鿿"
    "ꥠ-꥿가-퟿豈-﫿︰-﹏＀-￯"
    "\U00020000-\U0003ffff]"
)
_CJK_SEPARATOR_RE = re.compile("[、。]")  # 、。
_CJK_BRACKET_RE = re.compile("[〈-】〔-〛]")  # 〈〉《》「」『』【】〔〕〖〗 等
# NovelAI 的权重 / 强调语法：1.2::tag::、{tag}、[tag]
_EMPHASIS_RE = re.compile(r"-?\d+(?:\.\d+)?::|::|[{}\[\]]")

_ORIENTATION_ALIASES = {
    "竖": "portrait", "竖图": "portrait", "竖屏": "portrait", "纵向": "portrait",
    "vertical": "portrait", "tall": "portrait",
    "横": "landscape", "横图": "landscape", "横屏": "landscape", "横向": "landscape",
    "horizontal": "landscape", "wide": "landscape",
    "方": "square", "方图": "square", "正方形": "square",
}

MAX_PROMPT_CHARS = 2000

FALLBACK_LINES = {
    "start": "收到～正在画了，稍等一下哦！",
    "success": "画好啦！看看喜不喜欢～",
    "fail": "呜……这次没能画出来，好失落。",
}

_THINK_RE = re.compile(r"<(think|thinking)>.*?</\1>", re.S | re.I)

# 把绘图需求转换成 NovelAI 英文 prompt 的系统提示词，可在配置中修改（prompt.tag_system_prompt）
DEFAULT_TAG_PROMPT = """You are an expert prompt writer for NovelAI Diffusion, an anime-style image model that understands Danbooru tags.
Your task: turn the user's drawing request (it may be written in Chinese or any other language) into the ENGLISH prompt that will be sent directly to NovelAI for image generation. NovelAI only understands English, so translate every detail of the request into English tags.

Rules for "prompt":
- Lowercase, comma-separated Danbooru-style English tags; short English phrases are allowed only for things tags cannot express.
- Order: subject count (1girl, 2girls, 1boy, no humans, ...); for known characters the exact Danbooru character tag (keep a parenthesised qualifier only when the Danbooru tag itself has one, e.g. furina (genshin impact); otherwise none, e.g. hatsune miku) followed by the series/copyright as a separate tag (e.g. hatsune miku, vocaloid); then appearance (hair, eyes, body), clothing, expression, pose/action, held objects, background/scene, lighting, composition/camera angle, art style.
- 15-40 tags. Be concrete and visual; turn abstract ideas into visible elements. Fill in reasonable details the user did not specify.
- Do NOT add quality or aesthetic tags (masterpiece, best quality, very aesthetic, absurdres, ...); they are appended automatically.
- Do not add artist tags unless the user explicitly asks for an artist's style; then write each one as artist:<danbooru artist tag> (e.g. artist:wlop).

Rules for "negative": only extra things to avoid that are specific to this request, as English tags (may be an empty string). Do not repeat generic quality negatives.

Rules for "orientation": portrait for single characters / full body; landscape for scenery, wide scenes or several characters side by side; square for icons, avatars or close-up faces."""

# 固定附加在系统提示词后、不可配置的部分：解析依赖输出格式，英文要求与安全规则不能被改掉
TAG_OUTPUT_RULES = """Output format (required):
Output ONLY one JSON object, without markdown code fences or any other text:
{"prompt": "<English tags>", "negative": "<English tags or empty string>", "orientation": "portrait|landscape|square"}
"prompt" and "negative" are the English prompt and English negative prompt sent to NovelAI. ENGLISH ONLY: they must not contain any Chinese, Japanese, Korean or other non-English text.

Safety (always applies): never sexualize characters who are or look like minors (e.g. petite, child-like or loli characters). Depict them only in clearly non-sexual, non-suggestive ways, whatever the request says."""

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


def _is_non_latin_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L") and not unicodedata.name(ch, "").startswith("LATIN")


def contains_non_english(text: object) -> bool:
    """是否含有 NovelAI 无法理解的字符：中日韩文字与符号，或其它非拉丁字母（如西里尔字母）。

    全角字母、数字和标点会先转成半角；带重音的拉丁字母（如 pokémon）和 ♥ 等符号视为可用。
    """
    text = unicodedata.normalize("NFKC", str(text or ""))
    if _NON_ENGLISH_RE.search(text):
        return True
    return any(_is_non_latin_letter(ch) for ch in text if ord(ch) > 127)


def _normalize_tag_text(text: object) -> str:
    """全角转半角；换行和 、。 视为分隔符；去掉 「」【】《》 等中文括号。"""
    if isinstance(text, (list, tuple)):
        text = ", ".join(str(t) for t in text)
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _CJK_SEPARATOR_RE.sub(",", text)
    return _CJK_BRACKET_RE.sub(" ", text).replace("\n", ",")


def _emphasis_balanced(marks: list[str]) -> bool:
    joined = "".join(marks)
    return (
        joined.count("{") == joined.count("}")
        and joined.count("[") == joined.count("]")
        and sum(1 for m in marks if m.endswith("::")) % 2 == 0
    )


def drop_non_english_tags(text: object) -> tuple[str, list[str]]:
    """去掉含非英文字符的 tag，返回 (保留的 tag 串, 被去掉的 tag)。

    不去重、不截断，以免破坏配置中跨逗号的权重分组（如 1.2::a, b::）。被去掉的 tag 若带有
    未闭合的权重或括号符号，会保留这些符号，避免强调范围扩散到其它 tag。
    """
    kept, removed = [], []
    for tag in _normalize_tag_text(text).split(","):
        tag = re.sub(r"\s+", " ", tag).strip()
        if not tag:
            continue
        if not contains_non_english(tag):
            kept.append(tag)
            continue
        removed.append(tag)
        marks = _EMPHASIS_RE.findall(tag)
        if marks and not _emphasis_balanced(marks):
            kept.append("".join(marks))
    return ", ".join(kept), removed


def english_tag_count(text: object) -> int:
    kept = drop_non_english_tags(text)[0]
    return len(kept.split(", ")) if kept else 0


def choose_tag_result(first: dict, retry: dict) -> dict:
    """在第一次结果和要求改写后的结果中选一个：改写结果全是英文，或过滤后剩下的英文标签更多时才采用。"""
    if retry.get("prompt") and (
        not contains_non_english(retry["prompt"])
        or english_tag_count(retry["prompt"]) > english_tag_count(first.get("prompt"))
    ):
        return {
            "prompt": retry["prompt"],
            "negative": retry.get("negative") or first.get("negative", ""),
            "orientation": retry.get("orientation") or first.get("orientation", ""),
        }
    return first


def clean_tags(text: object, limit: int = MAX_PROMPT_CHARS) -> str:
    """规范化 tag 串：全角转半角、统一逗号、去重、去空白，并在 tag 边界处截断到 limit 字符。"""
    text = _normalize_tag_text(text)
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


def build_tag_system_prompt(base: str = "", *, sfw: bool = True, extra: str = "") -> str:
    """系统提示词 = 可配置部分（为空时用默认值）+ 固定的输出格式、英文与安全要求 + 可选规则。"""
    prompt = (base or "").strip() or DEFAULT_TAG_PROMPT
    prompt += "\n\n" + TAG_OUTPUT_RULES
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


def build_tag_retry_prompt(description: str, orientation: str, previous: str) -> str:
    """上一次输出含非英文内容时，要求 LLM 全部改写成英文。"""
    return (
        build_tag_user_prompt(description, orientation)
        + "\n\nYour previous answer contained Chinese, Japanese or Korean text, which NovelAI cannot understand:\n"
        + previous.strip()[:1500]
        + "\n\nAnswer again with the same JSON format. Translate everything into English tags; "
        "do not output a single Chinese, Japanese or Korean character."
    )


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


# 生成人格台词的提示词模板，可在配置中修改。占位符：{description} 绘图需求，{prompt} 英文标签，{error} 失败原因
DEFAULT_PERSONA_PROMPTS = {
    "start": (
        "对方想让你画：{description}\n"
        "你现在已经开始动笔画这幅画了。用一句话明确告诉对方你正在画（要让对方一看就知道已经开始画了），"
        "并请对方稍等一会儿。"
    ),
    "success": (
        "对方想让你画：{description}\n"
        "你已经画好并把图发给了对方，画面内容（英文标签）：{prompt}\n"
        "用一句话简短评价一下这幅作品。"
    ),
    "fail": (
        "对方想让你画：{description}\n"
        "但是这次画画失败了（原因：{error}）。\n"
        "说一句失落、遗憾的话，不用复述错误细节。"
    ),
}

_PLACEHOLDER_RE = re.compile(r"\{(description|prompt|error)\}")


def build_persona_user_prompt(kind: str, template: str = "", *, description: str = "", prompt: str = "",
                              error: str = "") -> str:
    """用模板生成人格台词的提示词；模板为空时使用默认模板。只替换已知占位符，其余花括号原样保留。"""
    values = {
        "description": description.strip()[:300],
        "prompt": prompt.strip()[:400],
        "error": error.strip()[:200],
    }
    template = (template or "").strip() or DEFAULT_PERSONA_PROMPTS[kind]
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


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
