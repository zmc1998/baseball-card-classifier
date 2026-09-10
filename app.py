"""
棒球卡属性识别 - Streamlit 应用

工艺(foil)和颜色(color)两个模型，可单独用也可组合用。
类别名称从模型文件里的 vocab 直接读取，不在这里手写，避免和训练时顺序不一致。
"""

import io
import os

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# 模型文件：放在本目录下即可被识别，缺一个也能只用另一个
FOIL_MODEL = "foil.pt"
COLOR_MODEL = "color.pt"
LEGACY_MODEL = "model.pt"          # 旧版单模型部署的文件名，作为工艺模型的回退

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

# 颜色的中文名 + 展示用色块
COLOR_DESC = {
    "sky_blue": ("天蓝", "#87CEEB"), "neon_green": ("荧光绿", "#39FF14"),
    "rose_gold": ("玫瑰金", "#B76E79"), "aqua": ("水蓝", "#00FFFF"),
    "teal": ("青绿", "#008080"), "navy": ("藏青", "#1B2A5E"),
    "blue": ("蓝", "#1E6FD9"), "purple": ("紫", "#7B2FBE"),
    "fuchsia": ("洋红", "#FF00A0"), "pink": ("粉", "#FF9CC8"),
    "burgundy": ("酒红", "#6E1423"), "red": ("红", "#D62828"),
    "orange": ("橙", "#F77F00"), "yellow": ("黄", "#FFD60A"),
    "gold": ("金", "#C9A227"), "green": ("绿", "#2A9D3F"),
    "black": ("黑", "#1A1A1A"), "white": ("白", "#F5F5F5"),
    "silver": ("银", "#C0C0C0"), "platinum": ("铂", "#D8D8D0"),
    "sepia": ("棕褐", "#8A6A4A"), "camo": ("迷彩", "#5A6B3B"),
    "rainbow": ("彩虹", "#FF6B6B"), "multi": ("多色", "#9B5DE5"),
}


class MultiHeadNet(nn.Module):
    """与 common/train.py 的定义保持一致。"""

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


@st.cache_resource(show_spinner=False)
def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch, tasks, vocab = ckpt["arch"], ckpt["tasks"], ckpt["vocab"]
    img_size = ckpt.get("img_size", 224)

    model = MultiHeadNet(arch, tasks, vocab)
    model.load_state_dict(ckpt.get("model") or ckpt["state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    meta = {"arch": arch, "task": tasks[0], "vocab": vocab,
            "img_size": img_size, "val_acc": ckpt.get("val_acc"), "path": path}
    return model, tf, meta


@st.cache_resource(show_spinner="加载模型...")
def load_all():
    """返回 {"foil": bundle, "color": bundle}，缺失的模型不出现在字典里。"""
    found = {}
    for want, candidates in (("foil", [FOIL_MODEL, LEGACY_MODEL]),
                             ("color", [COLOR_MODEL])):
        for p in candidates:
            if not os.path.exists(p):
                continue
            try:
                bundle = load_model(p)
            except Exception as exc:            # 文件损坏或结构不匹配
                st.warning(f"{p} 加载失败：{exc}")
                continue
            if bundle[2]["task"] == want:       # 按模型自报的任务归位
                found[want] = bundle
                break
    return found


def predict(bundle, image):
    model, tf, meta = bundle
    tensor = tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor)[meta["task"]], dim=1)[0]
    return probs


def swatch(name):
    """颜色类别的行内色块"""
    hexv = COLOR_DESC.get(name, ("", "#CCCCCC"))[1]
    return (f'<span style="display:inline-block;width:11px;height:11px;'
            f'border-radius:3px;background:{hexv};'
            f'border:1px solid rgba(128,128,128,.45);'
            f'margin-right:6px;vertical-align:-1px"></span>')


def label_of(task, name):
    if task == "color":
        cn = COLOR_DESC.get(name, ("", ""))[0]
        return f"{name}" + (f"（{cn}）" if cn else "")
    return name


def render_result(task, bundle, probs, top_n):
    _, _, meta = bundle
    classes = meta["vocab"][task]
    order = torch.argsort(probs, descending=True)
    best = classes[order[0]]
    best_p = float(probs[order[0]])
    second_p = float(probs[order[1]]) if len(order) > 1 else 0.0

    head = "🎨 颜色" if task == "color" else "✨ 工艺"
    st.markdown(f"##### {head}")

    if task == "color":
        st.markdown(
            f"<div style='font-size:1.5rem;font-weight:600;line-height:1.9'>"
            f"{swatch(best)}{label_of(task, best)}"
            f"<span style='font-size:1rem;font-weight:400;opacity:.65'>"
            f" &nbsp;{best_p:.1%}</span></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='font-size:1.5rem;font-weight:600;line-height:1.9'>"
            f"{best}<span style='font-size:1rem;font-weight:400;opacity:.65'>"
            f" &nbsp;{best_p:.1%}</span></div>",
            unsafe_allow_html=True)
        if FOIL_DESC.get(best):
            st.caption(f"特征：{FOIL_DESC[best]}")

    margin = best_p - second_p
    if best_p < 0.5:
        st.warning("置信度偏低，结果可能不可靠")
    elif margin < 0.15:
        st.warning(f"与第二候选 {label_of(task, classes[order[1]])} "
                   f"接近（差 {margin:.1%}），建议人工复核")

    for rank in range(min(top_n, len(classes))):
        idx = order[rank]
        p = float(probs[idx])
        txt = label_of(task, classes[idx])
        st.progress(p, text=f"{txt} — {p:.1%}")


# ---------------- 页面 ----------------
st.set_page_config(page_title="棒球卡属性识别", page_icon="🎴", layout="wide")

# 布局与投放区样式。两点说明：
# 1) file_uploader 只在自身区域接收拖放，页面其他位置的 drop 事件拿不到
#    （组件在 iframe 内），所以把投放区做大做显眼来提高命中率。
# 2) 左栏 sticky 固定、右栏独立滚动，避免左侧控件被结果列表推走。
st.markdown("""
<style>
/* 页面本身不滚动，滚动交给右栏 */
section.main > div.block-container{
    padding-top: 2.2rem;
    padding-bottom: 1rem;
    max-width: 100%;
}

/* 左栏：固定在视口顶部，内容超高时自己滚 */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child{
    position: sticky;
    top: 0;
    align-self: flex-start;
    max-height: calc(100vh - 3rem);
    overflow-y: auto;
    padding-right: .85rem;
    border-right: 1px solid rgba(130,130,140,.22);
    scrollbar-width: thin;
}

/* 右栏：结果区独立滚动 */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child{
    max-height: calc(100vh - 3rem);
    overflow-y: auto;
    padding-left: .4rem;
    scrollbar-width: thin;
}

/* 投放区：加大、虚线、hover 高亮 */
section[data-testid="stFileUploaderDropzone"]{
    min-height: 132px;
    border: 2px dashed rgba(130,130,140,.55);
    border-radius: 12px;
    background: rgba(130,130,140,.05);
    transition: border-color .15s, background .15s;
}
section[data-testid="stFileUploaderDropzone"]:hover{
    border-color: rgba(70,130,220,.85);
    background: rgba(70,130,220,.07);
}
section[data-testid="stFileUploaderDropzone"] > div{
    flex-direction: column;
    gap: .5rem;
}
div[data-testid="stFileUploaderDropzoneInstructions"]{
    align-items: center;
    text-align: center;
}

/* 已上传文件列表：限高并可滚，别把下方控件挤出视野 */
div[data-testid="stFileUploaderFileList"]{
    max-height: 168px;
    overflow-y: auto;
    scrollbar-width: thin;
}
</style>
""", unsafe_allow_html=True)

bundles = load_all()
if not bundles:
    st.error(
        "没有找到可用的模型文件。\n\n"
        f"请把训练好的模型放到本目录：工艺模型命名为 `{FOIL_MODEL}`，"
        f"颜色模型命名为 `{COLOR_MODEL}`。"
    )
    st.stop()

st.title("🎴 棒球卡属性识别")

# 左：上传与设置；右：结果
left, right = st.columns([1, 1.9], gap="large")

with left:
    available = [t for t in ("foil", "color") if t in bundles]
    names = {"foil": "工艺 Foil", "color": "颜色 Color"}

    st.markdown("##### 识别项目")
    picked = []
    for t in available:
        acc = bundles[t][2]["val_acc"]
        suffix = f"（验证 {acc:.1%}）" if acc is not None else ""
        if st.checkbox(names[t] + suffix, value=True, key=f"use_{t}"):
            picked.append(t)
    missing = [t for t in ("foil", "color") if t not in bundles]
    if missing:
        st.caption("未加载：" + "、".join(names[t] for t in missing))

    st.markdown("##### 上传卡片")
    uploaded = st.file_uploader(
        "把图片拖到这里，或点击选择（支持一次多张）",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True)

    max_cls = max((len(bundles[t][2]["vocab"][t]) for t in picked), default=5)
    top_n = st.slider("显示候选数", 1, min(8, max_cls), 3)

    with st.expander("模型信息"):
        for t in available:
            m = bundles[t][2]
            st.markdown(f"**{names[t]}**")
            st.caption(
                f"{m['arch']} · {m['img_size']}×{m['img_size']} · "
                f"{len(m['vocab'][t])} 类"
                + (f" · 验证 {m['val_acc']:.1%}" if m["val_acc"] is not None else "")
                + f" · `{m['path']}`")

    with st.expander("可识别的类别"):
        for t in available:
            st.markdown(f"**{names[t]}**")
            for name in bundles[t][2]["vocab"][t]:
                if t == "color":
                    st.markdown(
                        f"<div style='line-height:1.7'>{swatch(name)}"
                        f"<span style='font-size:.82rem;opacity:.8'>"
                        f"{label_of(t, name)}</span></div>",
                        unsafe_allow_html=True)
                else:
                    d = FOIL_DESC.get(name, "")
                    st.caption(f"**{name}**" + (f" — {d}" if d else ""))

with right:
    if not picked:
        st.info("请在左侧至少勾选一个识别项目。")
    elif not uploaded:
        st.info("在左侧上传卡片图片开始识别。"
                "建议用正面、光线均匀、卡面占画面主体的照片。")
    else:
        for i, file in enumerate(uploaded):
            if i:
                st.divider()
            try:
                image = Image.open(io.BytesIO(file.getvalue()))
            except Exception as exc:
                st.error(f"{file.name}：无法读取图片（{exc}）")
                continue

            col_img, col_res = st.columns([1, 1.45], gap="medium")
            with col_img:
                st.image(image, caption=file.name, use_container_width=True)

            with col_res:
                results = {t: predict(bundles[t], image) for t in picked}

                # 两项都选时，先给出合成命名（与文件命名体系一致）
                if len(picked) > 1:
                    parts = []
                    for t in ("color", "foil"):
                        if t in results:
                            cls = bundles[t][2]["vocab"][t]
                            parts.append(cls[int(torch.argmax(results[t]))])
                    st.markdown("##### 组合结果")
                    st.code("+".join(parts), language=None)

                for j, t in enumerate(picked):
                    if j:
                        st.markdown("")
                    render_result(t, bundles[t], results[t], top_n)
