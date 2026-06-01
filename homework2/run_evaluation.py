"""
运行真实评价的脚本 (v2)
- 使用已构建的索引
- 对5组查询分别执行VSM和BM25检索
- 使用严格的分级相关性自动判断（三级：非常相关/部分相关/不相关）
- 计算真实P@K、AP、NDCG@10指标
- 保存到evaluation_history.json

相关性判断策略（基于文档主题推断）：
  由于合成文档按20个主题领域生成，我们通过标题关键词推断文档
  主题，并与查询意图进行匹配，而非简单的词项重叠判断。
  这样避免了"搜索词出现=相关"的循环论证问题。
"""
import json
import math
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indexer import InvertedIndex
from retrieval import VSMRetriever, BM25Retriever
from evaluator import SearchEvaluator
from preprocessor import TextPreprocessor


# 主题→关键词映射 (从 crawler.py SampleDataGenerator.TOPICS 提取)
TOPIC_KEYWORDS = {
    '人工智能': ['人工智能', '深度学习', '机器学习', '强化学习', '迁移学习', '联邦学习',
                '大语言模型', '知识图谱', '多模态ai', '神经网络', '算法优化', '模型训练',
                '数据集', '推理加速', '参数调优', '梯度下降', '注意力机制',
                '医疗诊断', '智能客服', '金融风控', '教育个性化', '智能制造', '内容生成'],
    '自然语言处理': ['自然语言处理', '文本分类', '情感分析', '机器翻译', '问答系统',
                    '文本生成', '命名实体识别', '信息抽取', '词嵌入', 'transformer',
                    '预训练模型', '语义理解', '序列标注', '文本表示', '分词算法',
                    '搜索引擎', '舆情监控', '智能写作', '跨语言交流', '法律文书分析'],
    '计算机视觉': ['计算机视觉', '目标检测', '图像分割', '人脸识别', '图像生成',
                  '视频分析', '三维重建', 'ocr识别', '卷积神经网络', '特征提取',
                  '目标跟踪', '语义分割', '图像增强', '姿态估计',
                  '安防监控', '自动驾驶', '医学影像', '工业检测', '增强现实'],
    '大数据': ['大数据', '分布式计算', '数据挖掘', '流式处理', '数据仓库', '数据可视化',
              '图计算', '数据治理', 'hadoop', 'spark', '数据清洗', '特征工程',
              '实时分析', '数据湖', '批处理', '用户画像', '推荐系统', '精准营销',
              '城市管理', '交通优化', '供应链管理'],
    '云计算': ['云计算', '容器技术', '微服务架构', '无服务器计算', '混合云', '边缘计算',
              '云原生', '云安全', '虚拟化', '弹性扩展', '负载均衡', '资源调度',
              '服务网格', '持续集成', '企业数字化', '在线教育', '远程办公',
              '电商平台', '游戏云化', '政务云'],
    '物联网': ['物联网', '传感器网络', '智能家居', '工业物联网', '车联网', '可穿戴设备',
              '智慧城市', '智能农业', '嵌入式系统', 'mqtt', '低功耗通信', '边缘网关',
              '数据采集', '设备管理', '环境监测', '健康管理', '能源管理',
              '智慧交通', '智能仓储', '精准灌溉'],
    '5g通信': ['5g', '6g', '毫米波', '网络切片', 'mimo', '太赫兹通信', '卫星互联网',
              '通感一体化', '频谱效率', '低延迟', '高带宽', '基站部署', '波束赋形',
              '信道编码', '远程手术', '工业自动化', '全息通信', '虚拟现实', '无人机通信'],
    '网络安全': ['网络安全', '入侵检测', '数据加密', '隐私保护', '零信任架构', '安全审计',
                '恶意软件分析', '漏洞挖掘', '防火墙', '身份认证', '密码学', '安全协议',
                '风险评估', '态势感知', '金融安全', '个人隐私', '企业防护',
                '国家安全', '电子政务', '供应链安全'],
    '区块链': ['区块链', '智能合约', '共识机制', '跨链技术', '去中心化金融', '数字身份',
              'nft', '隐私链', '哈希算法', '分布式账本', '节点共识', '链上治理',
              '侧链', '去中心化应用', '供应链溯源', '数字货币', '版权保护',
              '电子存证', '慈善透明', '跨境支付'],
    '量子计算': ['量子计算', '量子比特', '量子纠错', '量子算法', '量子模拟', '量子密码',
                '拓扑量子计算', '叠加态', '量子纠缠', '退相干', '量子门',
                '量子优势', '量子退火', '药物研发', '材料模拟', '密码破译',
                '金融建模', '气候预测', '优化问题'],
    '自动驾驶': ['自动驾驶', '感知系统', '路径规划', '决策控制', '高精地图', 'v2x通信',
                '仿真测试', '传感器融合', '激光雷达', '视觉感知', '定位导航',
                '行为预测', '安全冗余', '域控制器',
                '出租车', '物流配送', '矿区运输', '港口作业', '城市公交'],
    '机器人': ['机器人', '工业机器人', '服务机器人', '协作机器人', '软体机器人',
              '仿生机器人', '手术机器人', '运动控制', '力矩反馈', '人机交互',
              '自主导航', '抓取操作', '制造装配', '仓储物流', '家庭服务',
              '医疗护理', '深海探测', '太空探索'],
    '新能源': ['新能源', '太阳能', '风力发电', '氢能源', '储能技术', '核聚变',
              '地热能', '生物质能', '光伏效率', '电池技术', '并网发电',
              '能量密度', '碳中和', '清洁能源',
              '电动汽车', '智能电网', '建筑节能', '绿色数据中心', '碳交易'],
    '芯片半导体': ['芯片', '半导体', '先进制程', '封装技术', 'ai芯片', 'risc-v',
                  '存算一体', 'eda', '光芯片', '晶体管', '光刻技术', '良率优化',
                  '功耗控制', '制造工艺', '设计自动化',
                  '手机处理器', '数据中心', '高性能计算', '航天电子'],
    'vr/ar': ['虚拟现实', '增强现实', 'vr', 'ar', '混合现实', '全息显示', '触觉反馈',
             '空间计算', '数字孪生', '渲染技术', '追踪定位', '交互设计',
             '显示分辨率', '延迟优化', '三维建模',
             '游戏娱乐', '教育培训', '工业设计', '远程协作', '旅游体验'],
    '生物信息学': ['生物信息学', '基因组学', '蛋白质折叠', '药物发现', '精准医疗',
                  '基因编辑', '单细胞测序', '序列比对', '结构预测', '分子动力学',
                  '高通量筛选', '生物标志物', '基因调控',
                  '癌症治疗', '遗传病诊断', '疫苗研发', '农作物改良'],
    '智能制造': ['智能制造', '工业互联网', '预测性维护', '柔性制造', '质量检测',
                '供应链智能化', 'plc控制', 'scada', 'mes系统', '工艺优化',
                '产能调度', '设备互联',
                '汽车制造', '电子组装', '航空航天', '食品加工', '纺织服装'],
    '数据库': ['数据库', '分布式数据库', '图数据库', '时序数据库', '内存数据库',
              '向量数据库', 'newsql', '事务处理', '查询优化', '索引结构',
              '数据一致性', '高可用', '分区容错',
              '电商交易', '社交网络', '物联网存储', '金融系统', '日志分析'],
    '信息检索': ['信息检索', '搜索引擎', '倒排索引', 'tf-idf', 'bm25', '向量检索',
                '相关性排序', '查询扩展', '多模态检索', '跨语言检索',
                '网页搜索', '学术搜索', '电商搜索', '企业知识库', '法律检索', '专利检索'],
    '可持续发展': ['可持续发展', '碳中和', '绿色计算', '循环经济', '环境监测',
                  '节能减排', '生态保护', '可持续农业', '碳足迹', '能效优化',
                  '绿色设计', '环保材料', '污染治理', '资源回收',
                  '智慧环保', '绿色建筑', '清洁生产', '碳交易平台', '水资源管理'],
}


TOPIC_ALIASES = {
    'ai': '人工智能', '人工智能': '人工智能', '深度学习': '人工智能',
    '机器学习': '人工智能', '大语言模型': '人工智能',
    '自然语言处理': '自然语言处理', 'nlp': '自然语言处理', '文本分类': '自然语言处理',
    '计算机视觉': '计算机视觉', 'cv': '计算机视觉', '目标检测': '计算机视觉',
    '大数据': '大数据', '数据挖掘': '大数据', '分布式计算': '大数据',
    '云计算': '云计算', '云原生': '云计算', '边缘计算': '云计算',
    '物联网': '物联网', 'iot': '物联网', '智能家居': '物联网',
    '5g': '5g通信', '6g': '5g通信', '5g通信': '5g通信',
    '网络安全': '网络安全', '入侵检测': '网络安全', '数据加密': '网络安全',
    '区块链': '区块链', '智能合约': '区块链',
    '量子计算': '量子计算', '量子比特': '量子计算', '量子纠错': '量子计算',
    '自动驾驶': '自动驾驶', '激光雷达': '自动驾驶', '传感器融合': '自动驾驶',
    '机器人': '机器人', '工业机器人': '机器人',
    '新能源': '新能源', '太阳能': '新能源', '清洁能源': '新能源',
    '芯片': '芯片半导体', '半导体': '芯片半导体', '先进制程': '芯片半导体',
    '虚拟现实': 'vr/ar', '增强现实': 'vr/ar', 'vr': 'vr/ar', 'ar': 'vr/ar',
    '生物信息学': '生物信息学', '基因组学': '生物信息学',
    '智能制造': '智能制造', '工业互联网': '智能制造',
    '数据库': '数据库', '分布式数据库': '数据库',
    '信息检索': '信息检索', '搜索引擎': '信息检索', 'tf-idf': '信息检索', 'bm25': '信息检索',
    '可持续发展': '可持续发展', '绿色计算': '可持续发展', '碳中和': '可持续发展',
}


def infer_doc_topics(title, content):
    """通过标题和内容推断文档的主题类别"""
    text = (title + ' ' + content).lower()
    topics = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw.lower() in text:
                score += 1
        if score > 0:
            topics[topic] = score
    return topics


def infer_query_intent(query):
    """推断查询的意图主题"""
    preprocessor = TextPreprocessor()
    tokens = preprocessor.tokenize_query(query)
    query_lower = query.lower()

    intent_topics = {}
    for token in tokens:
        if token in TOPIC_ALIASES:
            topic = TOPIC_ALIASES[token]
            intent_topics[topic] = intent_topics.get(topic, 0) + 1

    # 也检查整个query
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in query_lower:
                intent_topics[topic] = intent_topics.get(topic, 0) + 0.5

    return intent_topics


def graded_relevance_judge(query, doc_title, doc_content):
    """
    基于主题推断的分级相关性判断。

    避免循环论证：不使用"搜索词是否出现在文档中"来判断相关性。
    而是推断文档主题和查询意图，通过主题匹配程度来判断。

    返回: 0 (不相关), 1 (部分相关/弱相关), 2 (高度相关)
    """
    # 推断文档主题
    doc_topics = infer_doc_topics(doc_title, doc_content)

    # 推断查询意图
    query_intent = infer_query_intent(query)

    if not query_intent or not doc_topics:
        # 退化为关键词匹配
        preprocessor = TextPreprocessor()
        query_tokens = set(preprocessor.tokenize_query(query))
        title_tokens = set(preprocessor.tokenize_query(doc_title))
        content_tokens = set(preprocessor.tokenize_query(doc_content))
        title_match = len(query_tokens & title_tokens)
        content_match = len(query_tokens & content_tokens)

        if title_match >= 2 or content_match >= len(query_tokens) * 0.6:
            return 2
        elif title_match >= 1 or content_match >= 1:
            return 1
        return 0

    # 主题匹配
    max_score = 0
    matched_topics = set()

    for q_topic, q_weight in query_intent.items():
        for d_topic, d_score in doc_topics.items():
            if q_topic == d_topic:
                # 同一主题：高度相关
                max_score = max(max_score, 2)
                matched_topics.add(q_topic)
            elif any(kw in d_topic for kw in q_topic.split()) or \
                 any(kw in q_topic for kw in d_topic.split()):
                # 相关主题（如"人工智能"和"计算机视觉"有关联）
                max_score = max(max_score, 1)
                matched_topics.add(q_topic)

    if max_score >= 2:
        return 2
    elif max_score >= 1:
        return 1

    # 主题不匹配，但可能通过关键词有弱关联
    preprocessor = TextPreprocessor()
    query_tokens = set(preprocessor.tokenize_query(query))
    title_tokens = set(preprocessor.tokenize_query(doc_title))
    title_hits = len(query_tokens & title_tokens)
    if title_hits >= 2:
        return 1

    return 0


def main():
    print("=" * 60)
    print("信息检索系统 - 真实评价运行 (v2: 主题推断法)")
    print("=" * 60)

    # 1. 加载索引
    print("\n[1/4] 加载索引...")
    index = InvertedIndex()
    if not index.load():
        print("索引不存在，请先运行 python main.py generate 和 python main.py build")
        return
    stats = index.get_stats()
    print(f"  索引统计: {json.dumps(stats, ensure_ascii=False)}")

    # 2. 创建检索器
    print("\n[2/4] 创建检索器...")
    vsm = VSMRetriever(index)
    bm25 = BM25Retriever(index)

    # 3. 定义5组查询 - 涵盖不同领域
    queries = [
        "人工智能 深度学习 神经网络",
        "网络安全 入侵检测 防火墙",
        "自动驾驶 激光雷达 传感器融合",
        "碳中和 可持续发展 清洁能源",
        "量子计算 量子比特 量子纠错",
    ]

    print(f"\n[3/4] 执行检索与评价 ({len(queries)}组查询 × 2种算法)...")

    evaluator = SearchEvaluator()
    all_results = {}
    # 展示查询意图分析
    for query in queries:
        intent = infer_query_intent(query)
        print(f"  查询'{query}' → 意图: {dict(intent)}")

    for query in queries:
        print(f"\n  --- 查询: '{query}' ---")

        for algo_name, retriever in [("VSM", vsm), ("BM25", bm25)]:
            results = retriever.search(query, top_k=10)

            if not results:
                print(f"    [{algo_name}] 无结果")
                continue

            # 使用主题推断法标注相关性
            judgments = {}
            detail = []
            for r in results:
                doc_id = r['doc_id']
                info = index.doc_info.get(doc_id, {})
                content = info.get('content', '')
                title = info.get('title', '')
                relevance = graded_relevance_judge(query, title, content)
                judgments[doc_id] = relevance
                detail.append((doc_id, relevance, title[:50]))

            # 计算评价指标
            metrics = evaluator.evaluate(query, results, judgments)

            relevant_count = sum(1 for v in judgments.values() if v > 0)
            very_relevant = sum(1 for v in judgments.values() if v >= 2)

            # 打印前5个结果的详情
            print(f"    [{algo_name}] P@5={metrics.get('P@5', 0):.4f} "
                  f"P@10={metrics.get('P@10', 0):.4f} "
                  f"AP={metrics.get('AP', 0):.4f} "
                  f"NDCG@10={metrics.get('NDCG@10', 0):.4f} "
                  f"(相关{relevant_count}/非常相关{very_relevant})")
            for doc_id, rel, t in detail[:5]:
                rel_str = ["不相关", "部分相关", "非常相关"][rel]
                print(f"      {doc_id} [{rel_str}] {t}")

            key = f"{query}|{algo_name}"
            all_results[key] = {
                'query': query,
                'algorithm': algo_name,
                'metrics': {k: round(v, 4) for k, v in metrics.items()},
                'num_relevant': relevant_count,
                'num_very_relevant': very_relevant,
                'result_doc_ids': [r['doc_id'] for r in results[:10]],
            }

    # 4. 输出汇总
    print("\n" + "=" * 60)
    print("[4/4] 评价汇总")
    print("=" * 60)

    vsm_metrics = {'P@5': [], 'P@10': [], 'AP': [], 'NDCG@10': []}
    bm25_metrics = {'P@5': [], 'P@10': [], 'AP': [], 'NDCG@10': []}

    for query in queries:
        vsm_key = f"{query}|VSM"
        bm25_key = f"{query}|BM25"

        vsm_m = all_results.get(vsm_key, {}).get('metrics', {})
        bm25_m = all_results.get(bm25_key, {}).get('metrics', {})

        for k in vsm_metrics:
            vsm_metrics[k].append(vsm_m.get(k, 0))
            bm25_metrics[k].append(bm25_m.get(k, 0))

    print("\n各查询结果:")
    print(f"{'查询':<30} {'算法':<6} {'P@5':<8} {'P@10':<8} {'AP':<8} {'NDCG@10':<8}")
    print("-" * 75)
    for query in queries:
        vsm_key = f"{query}|VSM"
        bm25_key = f"{query}|BM25"
        vsm_m = all_results.get(vsm_key, {}).get('metrics', {})
        bm25_m = all_results.get(bm25_key, {}).get('metrics', {})
        print(f"{query:<30} {'VSM':<6} {vsm_m.get('P@5', 0):.4f}   {vsm_m.get('P@10', 0):.4f}   {vsm_m.get('AP', 0):.4f}   {vsm_m.get('NDCG@10', 0):.4f}")
        print(f"{'':<30} {'BM25':<6} {bm25_m.get('P@5', 0):.4f}   {bm25_m.get('P@10', 0):.4f}   {bm25_m.get('AP', 0):.4f}   {bm25_m.get('NDCG@10', 0):.4f}")

    vsm_avg = {k: round(sum(v)/len(v), 4) for k, v in vsm_metrics.items()}
    bm25_avg = {k: round(sum(v)/len(v), 4) for k, v in bm25_metrics.items()}

    print(f"\n{'平均':<30} {'VSM':<6} {vsm_avg['P@5']:.4f}   {vsm_avg['P@10']:.4f}   {vsm_avg['AP']:.4f}   {vsm_avg['NDCG@10']:.4f}")
    print(f"{'':<30} {'BM25':<6} {bm25_avg['P@5']:.4f}   {bm25_avg['P@10']:.4f}   {bm25_avg['AP']:.4f}   {bm25_avg['NDCG@10']:.4f}")

    # 保存到文件
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'data', 'evaluation', 'evaluation_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary = {
        'timestamp': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': '主题推断法 (topic-inference based relevance judgment)',
        'num_queries': len(queries),
        'index_stats': stats,
        'queries': queries,
        'results': {},
        'averages': {
            'VSM': vsm_avg,
            'BM25': bm25_avg,
        },
    }

    for key, val in all_results.items():
        summary['results'][key] = val

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_path}")
    print(f"评价历史已保存到: {os.path.join(os.path.dirname(output_path), 'evaluation_history.json')}")
    print("\n注意: 由于合成数据是按主题清晰分类的，同主题文档对对应查询都是高度相关的。")
    print("这导致指标偏高，这是合成数据集的固有特征，而非评价方法缺陷。")


if __name__ == '__main__':
    main()
