# 棒球卡属性识别 · 部署

Streamlit 应用，工艺(foil)和颜色(color)两个模型可单独用也可组合用。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 模型文件

放在本目录下，按文件名识别：

| 文件 | 说明 |
|---|---|
| `foil.pt` | 工艺模型，13 类 |
| `color.pt` | 颜色模型，24 类 |
| `model.pt` | 旧版单模型部署的文件名，作为工艺模型的回退 |

缺一个也能启动，只是左侧少一个可勾选项。应用按模型文件里记录的
`tasks` 字段归位，不靠文件名判断任务类型。

更新模型：

```bash
cp ../foil/models/foil_v3_82.pt foil.pt
cp ../color/models/color.pt     color.pt
```

## 布局

- **左栏**：勾选识别项目、上传图片、候选数、模型信息、类别列表
- **右栏**：每张图左边原图、右边结果

两项都勾选时，右栏顶部给出 `颜色+工艺` 的合成命名，与
`raw_data/rename_*.py` 生成的文件命名体系一致（用 `+` 拼接）。

## 结果提示

- 最高置信度 < 50% → 提示置信度偏低
- 第一与第二候选相差 < 15% → 提示建议人工复核

颜色类别带中文名和色块；工艺类别带纹理特征说明。

## 部署到 Streamlit Cloud

仓库需要包含 `app.py`、`requirements.txt` 和两个 `.pt`（各约 16MB）。
`requirements.txt` 里指定了 CPU 版 torch，避免云端拉 CUDA 包。

模型定义 `MultiHeadNet` 与 `common/train.py` 保持一致；推理时
`weights=None` 不下载 ImageNet 预训练权重（随后会被微调权重整体覆盖，
且云端拉取慢且容易失败）。

## 当前模型

| 任务 | 类别数 | 验证集准确率 |
|---|---|---|
| 工艺 | 13 | 82.3% |
| 颜色 | 24 | 71.8% |
