# 腾讯云轻量服务器部署指南

> 前提：已购 2核4G 北京实例（Ubuntu 24.04 + Docker 预装），防火墙已放行 TCP 8765。
> 全程约 20 分钟。命令在**服务器上**执行（SSH 登录后）。

---

## Step 0：SSH 登录 + 环境检查

```bash
ssh root@<你的公网IP>
# 密码登录（站内信里的密码）或已配好密钥免密
```

登录后检查 Docker 与镜像加速：

```bash
docker --version
# 应输出 Docker version 29.x

cat /etc/docker/daemon.json 2>/dev/null || echo "no daemon.json"
```

**看 daemon.json 输出**：
- 如果里面有 `"registry-mirrors"` 字段 → 已有加速，跳到 Step 1
- 如果没有（或输出 no daemon.json）→ 执行下面这段配置腾讯云内网加速：

```bash
cat > /etc/docker/daemon.json << 'EOF'
{
  "registry-mirrors": ["https://mirror.ccs.tencentyun.com"]
}
EOF
systemctl restart docker
docker info | grep -A2 "Registry Mirrors"
```

## Step 1：拉代码

```bash
cd /root
git clone https://github.com/thu25li/isaac-lab-rl-copilot-agent.git
cd isaac-lab-rl-copilot-agent
```

（Ubuntu 24.04 自带 git；若提示没有：`apt update && apt install -y git`）

## Step 2：写密钥文件（env-file，权限 600）

```bash
cat > /root/isaac.env << 'EOF'
DEEPSEEK_API_KEY=<你的真实 DeepSeek key>
AGENT_API_KEY=<自选一个强随机串>
PUBLIC_BASE_URL=http://<你的公网IP>:8765
HOST=0.0.0.0
PORT=7860
SKILL_ROOT=/app
FILES_DIR=/app/agent_server/logs/files
EOF
chmod 600 /root/isaac.env
```

⚠️ **把 `<你的公网IP>` 替换成真实 IP**（PUBLIC_BASE_URL 用于拼接附件下载 URL，必须是清小搭能访问的地址，不带 `/v1` 不带尾斜杠）。

## Step 3：构建镜像（5-10 分钟，apt/pip 走清华源）

```bash
docker build -t isaac-agent .
```

## Step 4：启动容器（开机自启 + 崩溃自动重启）

```bash
docker run -d \
  --name isaac-agent \
  --restart=always \
  -p 8765:7860 \
  --env-file /root/isaac.env \
  isaac-agent
```

## Step 5：验证

```bash
# 1. 容器状态（STATUS 应为 Up）
docker ps --filter name=isaac-agent

# 2. 本机健康检查
curl http://localhost:8765/health
# 预期：status ok，skill_modules loaded 9/9，deepseek api_key_present true

# 3. 鉴权验证（无 key 应 401）
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/v1/models
# 预期：401

# 4. 带 key 验证
curl -H "Authorization: Bearer <你的 AGENT_API_KEY，与 isaac.env 一致>" \
     http://localhost:8765/v1/models
# 预期：{"object":"list","data":[...]}
```

然后在你**自己电脑**的浏览器访问（外部可达性验证）：

```
http://<你的公网IP>:8765/health
```

能看到同样的 JSON = 部署成功。

## Step 6：真实对话测试（花一次 DeepSeek 调用）

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -H "Authorization: Bearer <你的 AGENT_API_KEY，与 isaac.env 一致>" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"辅助模式\n帮我生成四足机器人前进的 reward"}]}'
```

预期返回里 `x_soda.attachments` 含 `reward_generated.py` + `reward_weights.png`。把其中一个 `fileUrl` 在浏览器打开，能下载 = 附件链路通。

## 接入清小搭

| 字段 | 值 |
|------|------|
| API 地址 | `http://<你的公网IP>:8765/v1` |
| API 密钥 | `<你的 AGENT_API_KEY，与 isaac.env 一致>` |
| 鉴权方式 | Bearer Token |
| 模型名 | `isaac-lab-rl-copilot` |

---

## 常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| build 时拉不动 python:3.11-slim | 没配镜像加速 | 回 Step 0 配 registry-mirrors |
| build 时 apt/pip 慢 | — | Dockerfile 已换清华源，不应出现；若出现检查是否用了旧版 Dockerfile（git pull） |
| /health 不通（外部） | 防火墙没放行 8765 | 控制台 → 防火墙 → TCP 8765 / 0.0.0.0/0 |
| 容器反复重启 | env 缺 DEEPSEEK_API_KEY / AGENT_API_KEY | `docker logs isaac-agent` 看报错，检查 /root/isaac.env |
| 对话报 DeepSeek 401 | key 错/余额尽 | 查 https://platform.deepseek.com/usage |
| 附件 fileUrl 打不开 | PUBLIC_BASE_URL 填错 | 检查 env 文件里 IP 与端口（:8765），改后 `docker rm -f isaac-agent` 重跑 Step 4 |

## 日常运维

```bash
docker logs -f isaac-agent        # 实时日志
docker restart isaac-agent        # 重启
# 更新代码：
cd /root/isaac-lab-rl-copilot-agent && git pull
docker build -t isaac-agent . && docker rm -f isaac-agent
# 然后重跑 Step 4 的 docker run
```
