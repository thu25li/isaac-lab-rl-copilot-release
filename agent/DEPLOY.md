# 部署到 Hugging Face Spaces 完整指南

> 5 个步骤，约 30 分钟完成部署。永久免费，国内可访问。

---

## 前置准备

- ✅ Hugging Face 账号（[注册](https://huggingface.co/join)，免费）
- ✅ DeepSeek API key（[获取](https://platform.deepseek.com/api_keys)）
- ✅ 本地能跑 `pytest tests/` 通过

---

## Step 1：创建 HF Space

1. 访问 https://huggingface.co/new-space
2. 填写：
   - **Space name**: `isaac-lab-rl-copilot`（或任意名）
   - **License**: MIT
   - **SDK**: **Docker**（必须选 Docker，因为我们要自定义 Dockerfile）
   - **Hardware**: **CPU basic (16 GB RAM, 2 vCPU)** — 免费层够用
   - **Visibility**: Public（评委能访问）
3. 点击 **Create Space**

创建后，HF 会给你一个公网 URL：`https://<your-name>-isaac-lab-rl-copilot.hf.space`

---

## Step 2：克隆 HF Space 仓库 + 拷贝文件

HF Space 本质是个 git 仓库。本地克隆后，把项目文件拷进去。

```bash
# 1. 克隆刚创建的空 Space（HF 会要求用户名 + Access Token 作密码）
git clone https://huggingface.co/spaces/<your-name>/isaac-lab-rl-copilot hf-space
cd hf-space

# 2. 把项目主体拷进来（从 skillpro 工作目录）
SOURCE="D:/桌面/skillpro/isaac-lab-rl-copilot"

cp -r "$SOURCE/agent_server" .
cp -r "$SOURCE/scripts" .
cp -r "$SOURCE/resources" .
cp -r "$SOURCE/templates" .
cp -r "$SOURCE/references" .
cp -r "$SOURCE/tests/test_data" tests/  # 注意：tests/test_data 整体拷

# 3. 拷 Dockerfile + .dockerignore + README.md（HF 配置）
cp "$SOURCE/deploy/hf-spaces/Dockerfile" .
cp "$SOURCE/deploy/hf-spaces/.dockerignore" .
cp "$SOURCE/deploy/hf-spaces/README.md" .  # 覆盖默认 README
```

最终 Space 仓库结构：
```
hf-space/
├── Dockerfile
├── .dockerignore
├── README.md             ← HF Space 配置（frontmatter）
├── agent_server/
│   ├── main.py
│   ├── core/
│   ├── tools/
│   └── tests/
├── scripts/
├── resources/
├── templates/
├── references/
└── tests/test_data/
```

---

## Step 3：配置 Secrets

在 HF Space 网页 → **Settings** → **Variables and secrets**：

**Secrets（加密，不会暴露）**：

| Name | Value |
|------|-------|
| `DEEPSEEK_API_KEY` | `sk-...`（你的真实 key） |
| `AGENT_API_KEY` | `sk-isaac-lab-rl-copilot-hf`（自定义，任意字符串） |
| `PUBLIC_BASE_URL` | `https://<your-name>-isaac-lab-rl-copilot.hf.space` |

**Variables（可选，明文）**：

| Name | Value |
|------|-------|
| `DEEPSEEK_MODEL` | `deepseek-chat`（默认值，可不配） |

⚠️ **注意**：`PUBLIC_BASE_URL` 是 Space URL（不带 `/v1`），不是 base_url。它是用来拼接附件下载 URL 的。

---

## Step 4：Commit + Push（触发自动构建）

```bash
cd hf-space
git add .
git commit -m "Initial deploy: Isaac Lab RL Co-pilot agent"
git push
```

Push 后，HF Space 网页会显示构建日志：
1. **Building**（2-5 分钟）—— Docker build
2. **Running**（1 分钟）—— 容器启动
3. **Ready** —— 显示访问 URL

如果构建失败，日志会指出问题。

---

## Step 5：验证部署

部署成功后，在浏览器访问：

```bash
# 1. 健康检查（应返回详细 components 字段）
https://<your-name>-isaac-lab-rl-copilot.hf.space/health

# 2. 列出模型（带 Bearer token）
curl -H "Authorization: Bearer sk-isaac-lab-rl-copilot-hf" \
     https://<your-name>-isaac-lab-rl-copilot.hf.space/v1/models

# 3. 完整对话
curl -X POST \
     -H "Authorization: Bearer sk-isaac-lab-rl-copilot-hf" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"教学模式\n帮我生成四足前进 reward"}]}' \
     https://<your-name>-isaac-lab-rl-copilot.hf.space/v1/chat/completions
```

预期：
- `/health` 返回 9 个 skill 模块全 loaded
- 对话返回 reward.py + reward_weights.png + sim2real_radar.png + mode_teaching.png（4 个 attachment）
- attachment 的 `fileUrl` 形如 `https://<your-name>-.../files/<uuid>.py`

---

## 接入清小搭

部署成功后，在清小搭"创建智能体"→"标准协议接入"填：

| 字段 | 值 |
|------|------|
| **API 地址** | `https://<your-name>-isaac-lab-rl-copilot.hf.space/v1` |
| **API 密钥** | `sk-isaac-lab-rl-copilot-hf`（你在 Secrets 配的 AGENT_API_KEY） |
| **鉴权方式** | Bearer Token |
| **模型名** | `isaac-lab-rl-copilot` |

填好后让清小搭自动探测，4 项应全过。

---

## 维护

### 更新代码后重新部署

```bash
cd hf-space
# 修改文件...
git add .
git commit -m "fix: ..."
git push  # 触发自动重新构建
```

### 查看 logs

HF Space 网页 → **Logs** 标签，实时看 stdout/stderr。

### 资源监控

HF Spaces 免费层：
- 16GB RAM、2 vCPU
- 持续运行无限制（不休眠）
- 流量不限

如果 free tier 不够（极少情况），升级到 16GB GPU（$0.60/h）等付费档。

---

## 常见问题

### 构建失败：matplotlib 字体错误

确保 Dockerfile 安装了 `fonts-noto-cjk`。`mode_card.py` 的 `_setup_cjk_font()` 会自动检测。

### 部署成功但调用 401

检查 `AGENT_API_KEY` Secret 是否配置正确，调用时 `Authorization: Bearer` 后是否完全匹配。

### attachment URL 拉不到

确认 `PUBLIC_BASE_URL` Secret 是完整 URL（包含 `https://`），不带 `/v1`，不带尾斜杠。

### DeepSeek 余额耗尽

登录 https://platform.deepseek.com/usage 查看余额。免费额度用完后按 ¥1/M token 计费。

### Space 自动休眠？

Docker Space **不休眠**（与 Gradio/Streamlit 的 free tier 不同）。但 HF 政策可能变化，长时间无访问可能进入低频保活模式（首次唤醒 5-10s）。

---

## 备份：本地 Docker 构建（不部署到 HF）

```bash
cd hf-space
docker build -t isaac-agent .
docker run -d --name isaac-agent \
    -p 7860:7860 \
    --env DEEPSEEK_API_KEY=sk-... \
    --env AGENT_API_KEY=sk-local-test \
    --env PUBLIC_BASE_URL=http://localhost:7860 \
    isaac-agent

# 测试
curl http://localhost:7860/health
```

适合本地验证 Dockerfile 是否正确，再 push 到 HF。
