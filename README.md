# 棒球卡箔纹识别

上传棒球卡图片，识别箔纹工艺（Refractor / X-fractor / Mojo Refractor 等）。

## 文件说明

```
app.py             Streamlit 应用
requirements.txt   依赖
model.pt           训练好的模型（需自己放进来）
```

类别名称从 `model.pt` 内部的 `vocab` 读取，**不需要在代码里手写**，
所以换模型时只替换 `model.pt` 即可，不用改 `app.py`。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

打开 http://localhost:8501

## 部署到 Streamlit Cloud

1. **准备模型文件**

   把要用的模型复制成 `model.pt`，和 `app.py` 放同一层：

   ```bash
   cp ../train/ft_v2.pt model.pt
   ```

2. **推到 GitHub**

   ```bash
   git init
   git add app.py requirements.txt README.md model.pt
   git commit -m "deploy foil classifier"
   git branch -M main
   git remote add origin https://github.com/<你的用户名>/<仓库名>.git
   git push -u origin main
   ```

   模型约 16MB，在 GitHub 100MB 单文件限制内，**不需要 Git LFS**。

3. **在 Streamlit Cloud 部署**

   打开 https://share.streamlit.io → New app → 选仓库和分支 →
   Main file path 填 `app.py` → Deploy。

   如果三个文件放在仓库的 `deploy/` 子目录里，Main file path 就填 `deploy/app.py`。

## 常见问题

**部署后报找不到 model.pt**

`.gitignore` 里如果有 `*.pt` 会把模型挡掉。确认它真的被提交了：

```bash
git ls-files | grep model.pt
```

没有输出就说明没提交，用 `git add -f model.pt` 强制加入。

**依赖安装很慢或内存超限**

`requirements.txt` 第一行的 `--extra-index-url .../whl/cpu` 是必须的，
它让 pip 装 CPU 版 torch（约 200MB）。去掉这行会装带 CUDA 的版本（约 2GB），
Streamlit Cloud 免费额度装不下。

**换模型后类别不对**

不会发生：类别来自模型文件本身。但要注意不同模型的类别数可能不同，
例如 `model.pt` 只有 10 类，`ft_v2.pt` 有 13 类。侧边栏会显示当前模型的实际类别。
