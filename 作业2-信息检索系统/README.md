# 信息检索系统(作业2)

面向**真实新闻语料**的中文信息检索系统:多算法检索(BM25 / VSM / 混合)、PRF 查询扩展、
人工相关性评价(P@k / MAP / nDCG)、以及基于**中文 CLIP(Chinese-CLIP)** 的跨媒体检索(以文搜图 / 以图搜文)。
全流程**纯 CPU 可跑,无需 GPU**。

> 数据全部为爬虫真实抓取,**无任何编造/生成数据**。

> 📋 **验收 / 复现**:见 [`验收指南.md`](验收指南.md) —— 每一步操作对应一个作业需求与一个评分档位。

---

## 一、功能与评分点对应

| 模块 | 说明 | 入口 |
|------|------|------|
| 🔍 多算法检索 | **BM25**(Okapi, k1=1.5/b=0.75)、**VSM**(TF-IDF + 余弦)、**混合检索**(BM25+VSM+语义,RRF 融合) | `/` |
| 🧠 查询扩展 | **PRF 伪相关反馈**:取 Top 文档高权重词扩展查询,提升召回 | 检索页"查询扩展"开关 |
| 📈 人工评价 | 每条结果旁**打分按钮**(相关/部分相关/不相关)→ 实时算 **P@5 / P@10 / MAP / nDCG** | `/evaluate` |
| 🖼 多模态检索 | **中文 CLIP**(Chinese-CLIP，ViT-B-16 视觉塔 + 中文 RoBERTa 文本塔)中文**以文搜图**、上传图**以图搜文** | `/multimodal` |
| 📊 数据看板 | 语料规模、来源/类别分布可视化 | `/dashboard` |
| 🌱 可持续发展 | 系统对环境(低能耗)与社会(信息可达)可持续的贡献 | `/sustainability` |

---

## 二、技术架构

```
爬虫 (异步) ──→ 文档库 (data/documents) ──→ 倒排索引 (BM25/VSM)
   │                                          └→ 语义向量 (sentence-transformers)
   └─→ 配图下载 (data/images) ──→ VLM质检(剔除无关图) ──→ 中文CLIP 图像向量索引 (FAISS)

检索请求 ──→ 混合检索器 (RRF 融合 BM25 + VSM + 语义) ──→ 结果 + 人工评价
图文检索 ──→ 中文 CLIP 联合空间 ──→ 以文搜图 / 以图搜文
```

- **爬虫**:`aiohttp` 异步并发 + UA 轮换 + 单域限流(礼貌抓取)+ `readability` 正文抽取 + **SimHash 近似去重** + 断点续爬。
- **检索**:倒排索引一次构建持久化;混合检索用 **RRF(k=60)** 融合三路召回。
- **多模态**:中文原生 **Chinese-CLIP**(`OFA-Sys/chinese-clip-vit-base-patch16`,经 `transformers` 加载,512 维),在大规模中文图文对上训练,中文查询的图文对齐优于多语言通用 CLIP;向量入 **FAISS**(序列化存储,规避 Windows Unicode 路径问题)。保留 `open-clip` 多语言模型作回退后端(配置 `clip_backend` 可切换)。
- **图片质检(VLM 预处理)**:入库前先用**视觉大模型**(Qwen-VL,`qwen3.6-plus`)逐图判定是否为"有意义的新闻内容图",剔除 logo/广告横幅/二维码/纯文字快讯图/界面截图/装饰素材等无关图;判定结果缓存到 `data/vectors/vlm_verdicts.json` 复用(密钥仅取环境变量 `DASHSCOPE_API_KEY`,不入库,无密钥时自动降级为缓存+廉价过滤)。

---

## 三、安装与使用步骤

### 3.1 环境准备

1. **Python**:建议 3.10 及以上(本项目在 Python 3.13 上验证)。
2. **安装依赖**(纯 CPU,无需 GPU):
   ```bash
   pip install -r requirements.txt
   ```
   `sentence-transformers`、`open-clip`/`transformers` 等会自动拉取 `torch`、`opencv` 等,首次安装耗时较长。
3. **(可选)多模态增强密钥**:以文搜图的「AI 精排」与图库质检会调用通义千问视觉模型,密钥仅从环境变量读取:
   ```bash
   setx DASHSCOPE_API_KEY "你的密钥"    # Windows;设置后需重开终端使其生效
   ```
   不设置亦可运行——系统自动降级为余弦阈值,并复用随仓库附带的判定缓存。

### 3.2 启动系统

```bash
python run.py web
```
浏览器打开 **http://127.0.0.1:8090**。仓库已附带 `data/` 下构建好的索引、向量与语料,**开箱即用,无需先爬取或建索引**。

> ⚠️ 作业2 与作业3 默认均使用端口 **8090**,请**勿同时启动**;若端口被占用,先关闭占用进程(见 3.5)。

### 3.3 各功能操作步骤

**(1) 文本检索 —— 首页 `/`**
1. 在搜索框输入自然语言查询,如 `地震救援`、`新能源汽车`;
2. (可选)在高级选项中选择检索算法(BM25 / VSM / 混合检索)、勾选「查询扩展(PRF)」;
3. 点击「搜索」。结果按相关度由高到低排列,每条给出**相关度分数、标题、匹配片段(关键词高亮)、来源、日期与原文 URL**;
4. 对每条结果点 **相关 / 部分相关 / 不相关** 打分并提交,即可实时得到本次查询的 P@5 / P@10 / MAP / nDCG / MRR。

**(2) 以文搜图 / 以图搜文 —— `/multimodal`**
- **以文搜图**:输入中文描述(如 `地震`、`芯片`),可开启「AI 精排」开关,点击搜索;返回图片逐张带 AI 相关性判定徽章,并统计「候选 N → 相关 M、剔除 K」;
- **以图搜文**:上传一张图片,系统编码后返回与之相关的文档。

**(3) 检索评价汇总 —— `/evaluate`**:查看累计的人工相关性评价与各项指标(P@k / MAP / nDCG / MRR)。

**(4) 数据看板 —— `/dashboard`**:语料规模、来源与类别分布的可视化。

**(5) 可持续发展 —— `/sustainability`**:系统对环境与社会可持续发展贡献的说明。

### 3.4 从零构建语料与索引(可选)

```bash
python run.py crawl --max 800   # 异步爬取真实新闻到 data/documents
python run.py build             # 构建倒排索引(BM25 / VSM)
python run.py multimodal        # 下载配图、VLM 质检、构建中文 CLIP 图文向量索引
```
其中图库质检需要 `DASHSCOPE_API_KEY`(无密钥则跳过新图判定、保留廉价过滤结果)。

### 3.5 常见问题

- **端口 8090 被占用 / 与作业3 冲突**:`netstat -ano | findstr :8090` 查看占用进程 PID,`taskkill /F /PID <PID>` 关闭后重启;一次只运行一个系统。
- **页面已打开但结果或列表空白**:多为浏览器缓存所致,按 **Ctrl + Shift + R** 强制刷新。
- **多模态首次使用较慢或报下载错误**:`/multimodal` 与混合检索的语义召回需联网下载模型权重(Chinese-CLIP ~600MB、sentence-transformers 多语言模型);无网络时可提前在联网环境执行一次 `python run.py multimodal` 预热缓存,或仅演示文本检索(不影响基本功能与人工评价档)。
- **重新编译实验报告**:报告为 LaTeX,需以 **XeLaTeX** 编译(依赖 ctex 宏包):`xelatex 实验报告_作业2.tex`。

---

## 四、目录结构

```
作业2-信息检索系统/
├── run.py                  # 入口: crawl / build / multimodal / web / demo
├── requirements.txt
├── README.md
├── 实验报告_作业2.md
├── src/
│   ├── config.py           # 路径/算法/模型配置
│   ├── crawler/            # 异步爬虫(去重/限流/正文抽取)
│   ├── nlp/                # 中文分词与处理 (jieba)
│   ├── retrieval/          # 倒排索引 + BM25/VSM/混合检索 + PRF
│   ├── multimodal/         # CLIP 视觉编码 + 跨媒体检索
│   ├── storage/            # 文档库 / 向量库 (FAISS)
│   ├── evaluation/         # P@k / MAP / nDCG + 人工评价存储
│   └── web/                # FastAPI 后端 + 模板
├── scripts/                # build_index / build_multimodal / crawl / run_evaluation
└── data/
    ├── documents/          # 真实新闻文档 (JSON)
    ├── images/             # 真实配图
    ├── index/              # 倒排索引
    ├── vectors/            # CLIP / 语义向量索引
    └── evaluation/         # 人工评价记录
```

---

## 五、数据真实性

- `data/documents/` 内每篇文档均带原始 `url`、`source`、`date`,可逐条溯源至真实新闻页面。
- `data/images/` 为爬虫从新闻页**实际下载**的配图(下载时按 PNG/JPG 文件头校验)。
- 项目**不含任何数据生成器**,不存在编造文档。

---

## 六、可持续发展

详见 `/sustainability` 页面。核心:异步并发 + 礼貌抓取 + SimHash 去重 + 索引/向量**一次构建持久化复用** +
**纯 CPU 友好模型**,以更低算力/带宽/能耗达成同等检索效果;检索能力服务于公众信息可达性(社会可持续)。
