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

OpenBayes 模型部署强制使用 80 端口对外提供服务；工作空间默认暴露 8080。脚本统一处理两种场景，无需为部署改代码。

## 部署

参考官方文档完成部署，本示例的特殊点只有：

- 基础镜像选任一带 Python 3 + conda 的即可（**CPU 算力即可，无 GPU 需求**）
- 把本目录绑定到 `/openbayes/home`，确保根目录有 `start.sh`
- `requirements.txt` 会被自动 `pip install`，无需在 `start.sh` 里手动装

详细流程见：

- [Serving 快速上手](https://openbayes.com/docs/serving/getting-started/) — 创建 ServingVersion、绑定数据、启动部署的完整步骤
- [依赖管理](https://openbayes.com/docs/serving/dependencies/) — `requirements.txt` / `dependencies.sh` / `conda-packages.txt` 的执行顺序
- [创建模型部署版本工作流](https://openbayes.com/docs/serving/manage-servings/) — 版本管理、灰度、回滚

## 测试

### 1. 本地或工作空间内（端口 8080）

适合改完代码立即验证：

```bash
pip install -r requirements.txt
bash start.sh
# Starting Fake LLM Server on port 8080...
```

非流式 chat：

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "fake-gpt-3.5", "messages": [{"role": "user", "content": "hello"}]}'
```

流式 chat（注意 `-N` 关闭 buffer）：

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "fake-gpt-3.5", "stream": true, "messages": [{"role": "user", "content": "hello"}]}'
```

embeddings：

```bash
curl -X POST http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "fake-embedding-model", "input": "hello world"}'
```

健康检查：

```bash
curl http://localhost:8080/
# {"status": "Fake LLM Server is running"}
```

工作空间里如果想从外部访问 8080，参考 [暴露服务](https://openbayes.com/docs/gear/expose-service/)。

### 2. 部署后的线上版本（端口 80）

OpenBayes 提供两种测试方式，详见 [Serving 快速上手 - 在线测试](https://openbayes.com/docs/serving/getting-started/#%E5%9C%A8%E7%BA%BF%E6%B5%8B%E8%AF%95) 和 [命令行测试](https://openbayes.com/docs/serving/getting-started/#%E5%91%BD%E4%BB%A4%E8%A1%8C%E6%B5%8B%E8%AF%95)：

- **在线测试工具**：在模型部署详情页直接发请求，支持流式输出实时查看
- **curl 命令行**：用 OpenBayes 分配的 URL 替换 `localhost:8080` 即可，例如：

```bash
curl -X POST https://<serving-url>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "fake-gpt-3.5", "messages": [{"role": "user", "content": "hello"}]}'
```

## 行为说明

- chat 非流式：`uniform(1.2, 12)` 秒延迟后一次性返回。
- chat 流式：先等 `uniform(0.2, 2)` 秒（模拟 TTFT），按词流式返回，最后单独发送一个 `usage` chunk 和 `[DONE]`。
- embeddings：固定 10 维，按输入文本的 hash 做 seed，相同输入返回相同向量。
- token 数都是随机生成的，仅用于占位，不反映真实 tokenizer。
