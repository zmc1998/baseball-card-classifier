"""
棒球卡箔纹(Foil)识别 - Streamlit 应用

类别名称从模型文件里的 vocab 直接读取，不需要在这里手写，
避免和训练时的顺序不一致。
"""

import io
import json

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "model.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 各箔纹工艺的中文说明，仅用于展示；模型没有的类别不会出现
FOIL_DESC = {
    "Refractor": "基础衍射膜，倾斜时出现彩虹光带",
    "X-fractor": "网格状交叉纹理",
    "Atomic Refractor": "不规则斑块状纹理",
    "Mojo Refractor": "碎裂马赛克状",
    "Prism Refractor": "棱柱条纹",
    "Pulsar Refractor": "圆形波纹扩散",
    "Speckle Refractor": "细密颗粒点状",
    "Shimmer Refractor": "微闪细粉质感",
    "RayWave Refractor": "放射波浪线",
    "Lava Refractor": "熔岩流状",
    "Negative Refractor": "反色（底片效果）",
    "Sepia Refractor": "棕褐调",
    "SuperFractor": "大面积漩涡金箔，固定 1/1",
}


# ---------------- 模型定义（与 train.py 保持一致） ----------------
class MultiHeadNet(nn.Module):
    def __init__(self, arch, tasks, vocab, dropout=0.3):
        super().__init__()
        # 推理时不下载 ImageNet 预训练权重：随后会被微调权重整体覆盖，
        # 且云端拉取权重慢且容易失败
        if arch == "efficientnet_b0":
            m = models.efficientnet_b0(weights=None)
            feat = m.classifier[1].in_features
            m.classifier = nn.Identity()
        elif arch == "resnet18":
            m = models.resnet18(weights=None)
            feat = m.fc.in_features
            m.fc = nn.Identity()
        elif arch == "resnet50":
            m = models.resnet50(weights=None)
            feat = m.fc.in_features
            m.fc = nn.Identity()
        else:
            raise ValueError(arch)

        self.backbone = m
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, len(vocab[t])))
            for t in tasks
        })

    def forward(self, x):
        f = self.backbone(x)
        return {t: h(f) for t, h in self.heads.items()}


@st.cache_resource(show_spinner="加载模型...")
def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    arch = ckpt["arch"]
    tasks = ckpt["tasks"]
    vocab = ckpt["vocab"]
    img_size = ckpt.get("img_size", 224)

    model = MultiHeadNet(arch, tasks, vocab)

    state = ckpt.get("model") or ckpt.get("state_dict")
    model.load_state_dict(state)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    meta = {
        "arch": arch,
        "tasks": tasks,
        "vocab": vocab,
        "img_size": img_size,
        "val_acc": ckpt.get("val_acc"),
    }

    return model, tf, meta


def predict(model, tf, image, task):
    tensor = tf(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)[task]
        probs = torch.softmax(logits, dim=1)[0]

    return probs


# ---------------- 页面 ----------------
st.set_page_config(page_title="棒球卡箔纹识别", page_icon="🎴", layout="centered")

st.title("🎴 棒球卡箔纹识别")

try:
    model, tf, meta = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"找不到模型文件 `{MODEL_PATH}`。\n\n"
        "请把训练好的 `.pt` 文件重命名为 `model.pt` 放在本文件同目录下。"
    )
    st.stop()

task = meta["tasks"][0]
classes = meta["vocab"][task]

with st.sidebar:
    st.subheader("模型信息")
    st.write(f"**结构**：{meta['arch']}")
    st.write(f"**输入尺寸**：{meta['img_size']}×{meta['img_size']}")
    st.write(f"**识别任务**：{task}")
    st.write(f"**类别数**：{len(classes)}")
    if meta["val_acc"] is not None:
        st.write(f"**验证集准确率**：{meta['val_acc']:.1%}")

    st.divider()
    st.subheader("可识别的类别")
    for name in classes:
        desc = FOIL_DESC.get(name, "")
        st.caption(f"**{name}**" + (f" — {desc}" if desc else ""))

uploaded = st.file_uploader(
    "上传卡片图片",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="可以一次选多张",
)

if not uploaded:
    st.info("上传一张卡片图片开始识别。建议用正面、光线均匀、卡面占画面主体的照片。")
    st.stop()

top_n = st.slider("显示前几个候选", min_value=1, max_value=min(5, len(classes)), value=3)

for file in uploaded:
    st.divider()

    try:
        image = Image.open(io.BytesIO(file.getvalue()))
    except Exception as exc:
        st.error(f"{file.name}：无法读取图片（{exc}）")
        continue

    col_img, col_result = st.columns([1, 1.3])

    with col_img:
        st.image(image, caption=file.name, use_container_width=True)

    with col_result:
        probs = predict(model, tf, image, task)
        order = torch.argsort(probs, descending=True)

        best = classes[order[0]]
        best_prob = float(probs[order[0]])

        st.metric("识别结果", best, f"{best_prob:.1%}")

        desc = FOIL_DESC.get(best)
        if desc:
            st.caption(f"特征：{desc}")

        # 置信度低或前两名接近时给出提示
        second_prob = float(probs[order[1]]) if len(order) > 1 else 0.0
        margin = best_prob - second_prob

        if best_prob < 0.5:
            st.warning("置信度偏低，结果可能不可靠。")
        elif margin < 0.15:
            st.warning(f"与第二候选 {classes[order[1]]} 接近（差 {margin:.1%}），建议人工复核。")

        st.write("**候选**")
        for rank in range(top_n):
            idx = order[rank]
            st.progress(
                float(probs[idx]),
                text=f"{classes[idx]} — {float(probs[idx]):.1%}",
            )
