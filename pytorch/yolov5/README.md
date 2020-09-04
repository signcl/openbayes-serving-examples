# YOLOv5

依据 [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5) 魔改而来。

## 下载模型

执行以下命令下载所需模型：

```bash
bash fetch_models.sh
```

## 上传目录

按照[上传模型部署目录](https://openbayes.com/docs/bayesserving-quickstart/#%E4%B8%8A%E4%BC%A0%E5%88%B0%E6%95%B0%E6%8D%AE%E4%BB%93%E5%BA%93)中的介绍上传目录。

## 创建模型部署

按照[创建新的模型部署](https://openbayes.com/docs/bayesserving-quickstart/#创建-serving)创建一个新的模型部署。

## 测试

部署完成后可以用下面的命令进行测试:

```bash
curl "${ENDPOINT}" -X POST -H "Content-Type: application/json" -d @sample.json
```

其中 `ENDPOINT` 在「模型部署」概览页面上可以找到。

返回结果:

```json
[
  [
    [
      [
        500,
        235
      ],
      [
        757,
        710
      ]
    ],
    [
      [
        267,
        16
      ],
      [
        571,
        718
      ]
    ],
    [
      [
        713,
        190
      ],
      [
        1072,
        714
      ]
    ],
    [
      [
        24,
        260
      ],
      [
        343,
        720
      ]
    ]
  ],
  [
    "cat",
    "cat",
    "cat",
    "cat"
  ]
]
```