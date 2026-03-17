"""
信息检索系统 - 主入口
支持命令行操作和Web界面启动

用法:
    python main.py generate        生成样本数据（150篇中文科技文档）
    python main.py crawl [URL...]  从指定URL爬取网页数据
    python main.py build           构建倒排索引
    python main.py search <query>  命令行搜索
    python main.py evaluate        查看评价历史
    python main.py web             启动Web界面
    python main.py demo            一键演示（生成数据 → 构建索引 → 启动Web）
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, INDEX_DIR


def cmd_generate():
    """生成样本数据"""
    from crawler import SampleDataGenerator
    print("=" * 50)
    print("  步骤1: 生成样本数据")
    print("=" * 50)
    docs = SampleDataGenerator.generate(150)
    print(f"已生成 {len(docs)} 篇样本文档\n")


def cmd_crawl(urls=None):
    """爬取网页数据"""
    from crawler import WebCrawler
    print("=" * 50)
    print("  步骤1: 网络爬取")
    print("=" * 50)
    crawler = WebCrawler()
    if urls:
        crawler.crawl(seed_urls=urls)
    else:
        crawler.crawl()
    print()


def cmd_build():
    """构建倒排索引"""
    from indexer import InvertedIndex
    print("=" * 50)
    print("  步骤2: 构建倒排索引")
    print("=" * 50)
    index = InvertedIndex()
    index.build()
    stats = index.get_stats()
    print(f"\n索引统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()


def cmd_search(query, algorithm='bm25', use_expansion=False):
    """命令行搜索"""
    from indexer import InvertedIndex
    from retrieval import create_retriever, QueryExpander

    index = InvertedIndex()
    if not index.load():
        print("错误: 索引文件不存在，请先运行 'python main.py build'")
        return

    retriever = create_retriever(index, algorithm)

    if use_expansion:
        expander = QueryExpander(index)
        expanded = expander.expand(query)
        print(f"查询扩展: {' '.join(expanded)}")

    results = retriever.search(query, top_k=20)

    print("=" * 60)
    print(f"  查询: {query}")
    print(f"  算法: {algorithm.upper()}")
    print(f"  结果: {len(results)} 条")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"\n--- 第 {i} 条 (相关度: {r['score']}) ---")
        print(f"  标题: {r['title']}")
        print(f"  URL:  {r['url']}")
        print(f"  日期: {r['date']}")
        print(f"  摘要: {r['snippet_plain'][:100]}...")
        if r.get('images'):
            print(f"  图片: {r['images'][0]['alt']}")

    # 简易命令行评价
    if results:
        print("\n" + "=" * 60)
        answer = input("是否对结果进行人工评价？(y/n): ").strip().lower()
        if answer == 'y':
            _cli_evaluate(query, results)


def _cli_evaluate(query, results):
    """命令行下的人工评价"""
    from evaluator import SearchEvaluator

    judgments = {}
    print("\n请对每条结果评分 (0=不相关, 1=部分相关, 2=非常相关):")

    for i, r in enumerate(results[:10], 1):
        while True:
            try:
                score = input(f"  #{i} {r['title'][:30]}... [0/1/2]: ").strip()
                if score in ('0', '1', '2'):
                    judgments[r['doc_id']] = int(score)
                    break
                print("    请输入 0, 1 或 2")
            except (EOFError, KeyboardInterrupt):
                print("\n评价已取消")
                return

    evaluator = SearchEvaluator()
    metrics = evaluator.evaluate(query, results[:10], judgments)

    print("\n评价结果:")
    for k, v in metrics.items():
        print(f"  {k}: {round(v, 4)}")


def cmd_evaluate():
    """查看评价历史"""
    from evaluator import SearchEvaluator
    evaluator = SearchEvaluator()
    summary = evaluator.get_summary()

    print("=" * 50)
    print("  检索评价历史")
    print("=" * 50)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_web():
    """启动Web界面"""
    from app import app
    from config import FLASK_CONFIG
    print("=" * 50)
    print("  启动Web界面")
    print(f"  访问地址: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print("=" * 50)
    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug'],
    )


def cmd_demo():
    """一键演示"""
    print("=" * 60)
    print("  信息检索系统 - 一键演示")
    print("=" * 60)
    print()

    cmd_generate()
    cmd_build()
    cmd_web()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == 'generate':
        cmd_generate()
    elif command == 'crawl':
        urls = sys.argv[2:] if len(sys.argv) > 2 else None
        cmd_crawl(urls)
    elif command == 'build':
        cmd_build()
    elif command == 'search':
        if len(sys.argv) < 3:
            print("用法: python main.py search <查询词> [--algo vsm|bm25] [--expand]")
            return
        query = sys.argv[2]
        algo = 'bm25'
        expand = False
        if '--algo' in sys.argv:
            idx = sys.argv.index('--algo')
            if idx + 1 < len(sys.argv):
                algo = sys.argv[idx + 1]
        if '--expand' in sys.argv:
            expand = True
        cmd_search(query, algo, expand)
    elif command == 'evaluate':
        cmd_evaluate()
    elif command == 'web':
        cmd_web()
    elif command == 'demo':
        cmd_demo()
    else:
        print(f"未知命令: {command}")
        print(__doc__)


if __name__ == '__main__':
    main()
