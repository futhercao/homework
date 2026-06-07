"""
灾害信息抽取系统 (作业3) — 主入口

用法:
    python run.py crawl [--max 400]   异步爬取真实灾害新闻语料 (多来源)
    python run.py ocr                 对配图跑 EasyOCR, 结果回灌进文档
    python run.py extract             全量集成抽取 + 构建知识图谱 (生成 extractions.json)
    python run.py web                 启动 Web 界面 (抽取 / 图谱 / OCR / 抽取评价)
    python run.py demo                一键: crawl + ocr + extract + web

全流程纯 CPU 可跑, 无需 GPU。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')


def _run_script(name):
    import runpy
    runpy.run_path(os.path.join(SCRIPTS, name), run_name='__main__')


def cmd_crawl(args):
    print("=" * 60)
    print("  灾害语料爬取 — 异步抓取真实灾害新闻")
    print("=" * 60)
    sys.argv = ['crawl_disaster.py', '--max', str(args.max or 400)]
    _run_script('crawl_disaster.py')
    print("\n接下来运行: python run.py ocr  (可选) 然后 python run.py extract")


def cmd_ocr(args):
    print("=" * 60)
    print("  OCR — EasyOCR 识别配图文字, 回灌进文档")
    print("=" * 60)
    sys.argv = ['run_ocr.py']
    _run_script('run_ocr.py')


def cmd_extract(args):
    print("=" * 60)
    print("  全量集成抽取 + 知识图谱构建")
    print("=" * 60)
    _run_script('run_extraction.py')
    print("\n完成! 运行: python run.py web")


def cmd_web(args):
    print("=" * 60)
    print("  Web 服务模式")
    print("=" * 60)
    from src.web.app import run
    run()


def cmd_demo(args):
    print("=" * 60)
    print("  一键演示")
    print("=" * 60)
    cmd_crawl(args)
    cmd_ocr(args)
    cmd_extract(args)
    cmd_web(args)


def main():
    parser = argparse.ArgumentParser(description='灾害信息抽取系统 (作业3)')
    sub = parser.add_subparsers(dest='command')

    sp1 = sub.add_parser('crawl', help='爬取灾害语料')
    sp1.add_argument('--max', type=int, default=400)

    sub.add_parser('ocr', help='配图 OCR')
    sub.add_parser('extract', help='全量抽取 + 知识图谱')
    sub.add_parser('web', help='启动 Web 界面')
    sub.add_parser('demo', help='一键演示')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    globals()[f'cmd_{args.command}'](args)


if __name__ == '__main__':
    main()
