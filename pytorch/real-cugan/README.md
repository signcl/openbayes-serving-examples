# Real-CUGAN：一个使用百万级动漫数据进行训练的，结构与 Waifu2x 兼容的通用动漫图像超分辨率模型

来自 [bilibili/ailab](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) 一个动画超分辨率的样例。

## 创建模型部署

按照[创建新的模型部署](https://openbayes.com/docs/bayesserving-quickstart/#创建-serving)创建一个新的模型部署。

## 测试

部署完成后可以用下面的命令进行测试：

```bash
curl "${ENDPOINT}" \
    -X POST -H "Content-Type: application/octet-stream" \
    -d @sample.jpg
```

返回结果：

```json
[
    "comic_book",
    "cloak",
    "backpack",
    "abaya",
    "mask"
]
```
