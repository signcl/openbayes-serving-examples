# Fake LLM OpenAI-Compatible Server

一个 OpenAI 兼容的 fake LLM 服务，提供 `/v1/chat/completions`（含流式）和 `/v1/embeddings` 接口，用随机文本和随机延迟模拟真实 LLM 的响应。适合调试 LLM 网关、可观测性管线（Langfuse 等 tracing）以及 OpenBayes serving 流程本身，无需占用 GPU。

## 文件结构

```
fastapi/fake-llm-openai/
├── start.sh             # serving 入口，激活 base 并启动 uvicorn
├── fake_llm_server.py   # FastAPI 应用
├── requirements.txt     # fastapi / uvicorn
└── README.md
```

## 端口约定

`start.sh` 通过 `OPENBAYES_SERVING_PRODUCTION` 环境变量自动切换端口：

| 场景 | 环境 | 端口 | 触发条件 |
|------|------|------|----------|
| 开发 / 调试 | 工作空间（Workspace）或本地 | **8080** | `OPENBAYES_SERVING_PRODUCTION` 未设置 |
| 线上部署 | 模型部署（ServingVersion） | **80** | OpenBayes 自动注入 `OPENBAYES_SERVING_PRODUCTION` |

> OpenBayes 模型部署强制使用 80 端口对外提供服务；工作空间默认暴露 8080。脚本统一处理两种场景，无需为部署再改代码。

## 场景一：在工作空间 / 本地开发调试（8080）

适合反复修改 `fake_llm_server.py` 后立即验证。

```bash
# 在工作空间或本地仓库目录
pip install -r requirements.txt
bash start.sh
# Starting Fake LLM Server on port 8080...
```

测试请求（注意端口是 **8080**）：

```bash
# 非流式 chat
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "fake-gpt-3.5", "messages": [{"role": "user", "content": "hello"}]}'

# 流式 chat
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "fake-gpt-3.5", "stream": true, "messages": [{"role": "user", "content": "hello"}]}'

# embeddings
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "fake-embedding-model", "input": "hello world"}'

# 健康检查
curl http://localhost:8080/
```

工作空间内可以通过暴露 8080 端口的 frontend URL 访问，详见 [暴露服务](https://openbayes.com/docs/gear/expose-service/)。

## 场景二：发布为模型部署（80）

参考 [OpenBayes serving 快速上手](https://openbayes.com/docs/serving/getting-started/)：

1. 把本目录上传到数据集或工作空间，绑定到 `/openbayes/home`（必须包含 `start.sh`）。
2. 创建 ServingVersion，基础镜像选任一带 Python 3 + conda 的镜像（CPU 算力即可，无 GPU 需求）。
3. 部署后 OpenBayes 会自动：
   - `pip install -r requirements.txt` 安装依赖
   - 注入 `OPENBAYES_SERVING_PRODUCTION=1`
   - 执行 `start.sh`，服务监听 **80** 端口

部署完成后用 OpenBayes 分配的 URL 测试（不带端口，默认 80）：

```bash
curl -X POST https://<serving-url>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "fake-gpt-3.5", "messages": [{"role": "user", "content": "hello"}]}'
```

## 行为说明

- chat 非流式：`uniform(1.2, 12)` 秒延迟后一次性返回。
- chat 流式：先等 `uniform(0.2, 2)` 秒（模拟 TTFT），按词流式返回，最后单独发送一个 `usage` chunk 和 `[DONE]`。
- embeddings：固定 10 维，按输入文本的 hash 做 seed，所以同样输入返回同样向量。
- token 数都是随机生成的，仅用于占位，不反映真实 tokenizer。
