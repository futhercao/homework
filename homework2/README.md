# 信息检索系统

基于向量空间模型（VSM）和 BM25 算法的中文信息检索系统，支持网络爬取、倒排索引、相关性排序、人工评价和多媒体检索。

## 系统架构

```
homework2/
├── main.py              # 主入口（CLI命令行界面）
├── app.py               # Flask Web应用
├── config.py            # 全局配置
├── crawler.py           # 网络爬虫 + 样本数据生成器
├── preprocessor.py      # 文本预处理（jieba分词、停用词过滤）
├── indexer.py           # 倒排索引构建（TF-IDF）
├── retrieval.py         # 检索模块（VSM + BM25 + 查询扩展）
├── evaluator.py         # 检索结果评价（P@K/R@K/F1/MAP/NDCG）
├── requirements.txt     # Python依赖
├── templates/           # Web界面HTML模板
│   ├── base.html
│   ├── index.html
│   ├── results.html
│   ├── evaluate.html
│   └── about.html
├── static/
│   └── style.css
└── data/                # 运行时数据
    ├── documents/       # 爬取/生成的文档
    ├── index/           # 倒排索引文件
    └── evaluation/      # 评价结果
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键演示

```bash
python main.py demo
```

该命令会依次：生成150篇样本文档 → 构建倒排索引 → 启动Web界面。

### 3. 分步操作

```bash
# 生成样本数据（150篇中文科技新闻文档）
python main.py generate

# 或从指定网站爬取数据
python main.py crawl https://news.example.com/tech/

# 构建倒排索引
python main.py build

# 命令行搜索
python main.py search "人工智能"
python main.py search "深度学习" --algo vsm
python main.py search "自然语言处理" --algo bm25 --expand

# 启动Web界面
python main.py web
```

### 4. Web界面

启动后访问 http://127.0.0.1:5000，可以：

- 输入查询词进行搜索
- 选择 VSM 或 BM25 算法
- 启用查询扩展功能
- 对检索结果进行人工评价
- 查看评价指标（P@K, R@K, F1, MAP, NDCG）

## 核心算法

### 向量空间模型（VSM）

将文档和查询表示为 TF-IDF 加权的向量，使用余弦相似度计算匹配度：

- **TF(t,d)** = count(t in d) / |d|
- **IDF(t)** = log(N / df(t))
- **Cosine Similarity** = (q · d) / (|q| × |d|)

### BM25 算法（优化）

BM25 是经典概率检索模型，优于基本 VSM：

- **Score(q,d)** = Σ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1 - b + b × |d|/avgdl))
- k1 = 1.5, b = 0.75

### 查询扩展

基于词共现分析的查询扩展，自动找到与查询词高度关联的术语，提升检索召回率。

## 创新点

1. **双算法引擎**：同时实现 VSM 和 BM25 两种检索算法，用户可灵活切换对比
2. **查询扩展**：基于共现词分析的自动查询扩展
3. **多媒体信息检索**：索引文档中的图片元数据，搜索结果展示相关图片
4. **完善的评价体系**：支持 Precision@K、Recall@K、F1、MAP、NDCG
5. **现代化Web界面**：基于 Flask + Bootstrap 5 的响应式设计

## 对环境和社会可持续发展的影响

### 环境影响

- **节能爬取**：设置合理请求间隔，避免不必要的服务器负载
- **本地处理**：全部在本地完成，无需云端GPU集群
- **轻量算法**：TF-IDF和BM25计算效率高，能耗低
- **索引缓存**：一次构建反复使用

### 社会影响

- **知识普惠**：提升信息获取效率，促进教育公平
- **开源技术**：基于开源生态，促进技术共享
- **隐私保护**：本地运行，不上传用户数据
- **质量可控**：人工评价机制确保检索质量可追溯

## 技术栈

- Python 3.8+
- Flask (Web框架)
- jieba (中文分词)
- BeautifulSoup4 (网页解析)
- NumPy (数值计算)
- Bootstrap 5 (前端UI)
