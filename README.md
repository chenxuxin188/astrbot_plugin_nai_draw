# astrbot_plugin_nai_draw

让 AstrBot 借助 LLM 在聊天中调用 **NovelAI** 绘图的简化插件，兼容 OneBot11（aiocqhttp，如 NapCat、Lagrange）。

参考了 [astrbot_plugin_ppnai](https://github.com/dafeiwu666/astrbot_plugin_ppnai)，只保留了「聊天中让 Bot 画画」这一核心功能。

## 功能

- **对话中绘图**：插件向 Bot 的 LLM 注册了 `nai_draw` 函数工具。用户说「画一只在樱花树下睡觉的猫娘」之类的话时，Bot 会自己决定调用它。
- **按人格说话**：
  - 开始绘图前，按 Bot 当前人格的口吻发一句开画提示。
  - 成功后发送图片，再用人格口吻简短评价一下。
  - 失败时先把错误信息发出来，再说一句失落的话。
- **LLM 生成提示词**：用指定的 LLM 把自然语言需求转换成 NovelAI 的 Danbooru 风格标签，并自动选择竖图、横图或方图。
- **必定包含的正面 / 负面提示词**：在配置中设置，每次生成都会带上。
- **默认免费参数**：模型 `nai-diffusion-4-5-full`、832×1216、28 步、单张，Opus 会员不消耗 Anlas。开启「强制使用免费参数」后会自动把尺寸和步数限制在免费范围内（仅对 V4.5 及以下模型有效，V5 仍会消耗 Opus 用量条）。
- **管理员指令**：查询 Anlas 余额，查看绘图记录（包括每次的 Anlas 消耗）。

## 安装

在 AstrBot WebUI 的插件市场中通过仓库地址安装：

```
https://github.com/chenxuxin188/astrbot_plugin_nai_draw
```

也可以把仓库克隆到 `data/plugins/` 目录后重启 AstrBot。要求 AstrBot ≥ 4.10。

## 配置

在 WebUI 的插件配置页面中填写。

| 配置项 | 说明 |
| --- | --- |
| 管理员 ID 列表 | 可以使用管理员指令的 QQ 号 |
| NovelAI Token | 在 NovelAI 网页 *Account Settings → Get Persistent API Token* 获取，以 `pst-` 开头 |
| 绘图模型 | 默认 V4.5 Full。V5 会消耗 Opus 用量条，用尽后扣 Anlas |
| 默认画面方向 / 各方向尺寸 | 默认竖图 832×1216、横图 1216×832、方图 1024×1024 |
| 步数、CFG Scale、CFG Rescale、采样器、噪声调度 | NovelAI 生成参数，默认值与官网 V4.5 一致 |
| 强制使用免费参数 | 开启时把总像素限制在 1024×1024 以内、步数限制在 28 以内（V5 模型不受保护） |
| 代理地址 | 国内服务器通常需要，支持 http / socks5 |
| 生成提示词使用的 LLM | 留空则使用当前会话的模型 |
| 必定包含的正面 / 负面提示词 | 正面词追加在末尾（默认是 V4.5 Full 官方质量词），负面词放在最前面 |
| 仅生成全年龄内容 | 默认开启，要求 LLM 把露骨的需求改写为全年龄版本 |
| 生成人格台词使用的 LLM / 指定人格 | 留空则跟随当前会话的模型和人格 |
| 记录 Anlas 消耗 / 每页条数 / 最多保留条数 | 绘图记录相关 |

## 使用

### 普通用户

- 直接和 Bot 说想画什么，例如「帮我画一张初音未来在海边的横图」。群聊里需要按 AstrBot 的唤醒规则 @Bot 或使用唤醒前缀。
- 也可以用指令直接画：`/nai draw <描述>`

### 管理员

| 指令 | 作用 |
| --- | --- |
| `/nai account` | 查询订阅等级、到期时间、Anlas 余额（订阅 + 购买），以及当前参数是否免费 |
| `/nai audit [页码]` | 从新到旧查看绘图记录：时间、用户、群号、模型、尺寸、步数、种子、耗时、Anlas 消耗、需求和 Prompt、失败原因 |

记录保存在 `data/plugin_data/astrbot_plugin_nai_draw/audit.jsonl`。

## 注意事项

- **Bot 不调用绘图工具时，请检查：**
  - 当前使用的模型支持函数调用（Function Calling）。
  - WebUI 中 `nai_draw` 工具处于启用状态。
  - 如果人格设置了工具白名单，需要把 `nai_draw` 加进去。
- **重复回复**：工具调用后，主 LLM 不会再额外回复。但个别模型会在调用工具的同时输出一段文字，这段文字也会发出。
- **生成顺序**：同一个 NovelAI 账号同一时间只能生成一张图。插件会让请求排队依次生成，同一用户在上一张完成前不能发起新的请求。
- **Anlas 消耗记录**：在排队锁内查询生成前后的余额并计算差值，所以不会混入本插件其他请求的消耗。如果同一个 Token 同时在网页或其他程序中使用，差值仍可能包含那些消耗。
- **失败重试**：只有确定请求没有被 NovelAI 处理时才会自动重试，包括连接失败、账号并发锁定、502/503。超时等无法确定的情况不重试，避免重复扣费。
- **重载插件**：保存插件配置会触发重载。正在进行的绘图会被中断，Bot 会提示用户稍后重试。
- **WebUI 对话**：在 WebUI 自带的聊天页面中，Bot 会等绘图完成后再结束本轮回复。
- **指令冲突**：本插件使用 `/nai` 指令组。请不要与同样注册了 `nai` 指令的插件（例如 ppnai）同时启用。

## 开发

```bash
python -m pip install httpx pytest
python -m pytest tests
```

`nai_client.py`、`prompting.py`、`audit.py` 不依赖 AstrBot，可以单独测试。
