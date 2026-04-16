from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import json
import time
import random
import os


app = FastAPI(title="Fake LLM Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.options("/v1/chat/completions")
async def options_chat_completions():
    return {}


@app.options("/v1/embeddings")
async def options_embeddings():
    return {}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    model = body.get("model", "fake-gpt-3.5")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    ttft_delay = random.uniform(0.2, 2.0)
    total_delay = random.uniform(1.0, 10.0)

    prompt_content = " ".join([msg.get("content", "") for msg in messages if msg.get("content")])
    prompt_length = len(prompt_content)

    fake_response = f"This is a fake response to your message. Your input had {prompt_length} characters."

    fake_prompt_tokens = random.randint(50, 2000)
    fake_completion_tokens = random.randint(20, 1000)
    fake_total_tokens = fake_prompt_tokens + fake_completion_tokens

    usage = {
        "prompt_tokens": fake_prompt_tokens,
        "completion_tokens": fake_completion_tokens,
        "total_tokens": fake_total_tokens,
    }

    if stream:
        async def stream_generator():
            await asyncio.sleep(ttft_delay)

            words = fake_response.split(" ")

            remaining_delay = total_delay - ttft_delay
            if len(words) > 1:
                time_per_token = remaining_delay / len(words)
            else:
                time_per_token = 0

            for i, word in enumerate(words):
                text = word + (" " if i < len(words) - 1 else "")

                chunk = {
                    "id": f"chatcmpl-{time.time()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }

                yield f"data: {json.dumps(chunk)}\n\n"

                await asyncio.sleep(time_per_token)

            finish_chunk = {
                "id": f"chatcmpl-{time.time()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"

            usage_chunk = {
                "id": f"chatcmpl-{time.time()}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        await asyncio.sleep(ttft_delay + total_delay)

        response = {
            "id": f"chatcmpl-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": fake_response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

        return JSONResponse(response)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()

    embedding_delay = random.uniform(0.5, 2.5)
    await asyncio.sleep(embedding_delay)

    model = body.get("model", "fake-embedding-model")
    input_data = body.get("input", "")

    inputs = [input_data] if isinstance(input_data, str) else input_data

    embeddings_result = []

    total_tokens = random.randint(50, 8000)

    for i, text in enumerate(inputs):
        embedding_dim = 10
        rng = random.Random(hash(text) % 10000)
        fake_embedding = [rng.random() for _ in range(embedding_dim)]

        norm = sum(x * x for x in fake_embedding) ** 0.5
        fake_embedding = [x / norm for x in fake_embedding]

        embeddings_result.append(
            {
                "object": "embedding",
                "embedding": fake_embedding,
                "index": i,
            }
        )

    response = {
        "object": "list",
        "data": embeddings_result,
        "model": model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }

    return JSONResponse(response)


@app.get("/")
async def root():
    return {"status": "Fake LLM Server is running"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
