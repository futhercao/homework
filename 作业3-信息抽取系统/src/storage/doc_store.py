"""
文档存储层 — JSON文件 + SQLite索引
- 支持CRUD操作
- 支持按类别、日期、来源过滤
- JSON为数据源（可审计），SQLite为快速查询索引
"""
import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Optional

from src.config import DOC_DIR


class DocumentStore:
    """统一的文档存储管理"""

    def __init__(self, doc_dir=None, db_path=None):
        self.doc_dir = doc_dir or DOC_DIR
        self.db_path = db_path or os.path.join(self.doc_dir, 'metadata.db')
        os.makedirs(self.doc_dir, exist_ok=True)
        self._init_db()
        self._sync_if_stale()

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute('''CREATE TABLE IF NOT EXISTS docs (
            id TEXT PRIMARY KEY,
            title TEXT, url TEXT, date TEXT, category TEXT,
            source TEXT, content_length INTEGER,
            image_count INTEGER, crawl_time TEXT,
            is_cleaned INTEGER DEFAULT 0
        )''')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON docs(category)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_date ON docs(date)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON docs(source)')
        self.conn.commit()

    def exists(self, doc_id):
        cur = self.conn.execute('SELECT 1 FROM docs WHERE id=?', (doc_id,))
        return cur.fetchone() is not None

    def reindex_from_files(self):
        """以 doc_dir 下的 JSON 文件为权威数据源, 重建 SQLite 索引。
        灾害爬虫 (crawl_disaster.py) 直接落盘 JSON、不写 db, 故用此法保持索引与磁盘一致。"""
        import glob as _glob
        self.conn.execute('DELETE FROM docs')
        n = 0
        for fp in _glob.glob(os.path.join(self.doc_dir, '*.json')):
            try:
                doc = json.load(open(fp, 'r', encoding='utf-8'))
            except Exception:
                continue
            if not doc.get('id'):
                continue
            self.conn.execute('''INSERT OR REPLACE INTO docs
                (id, title, url, date, category, source, content_length,
                 image_count, crawl_time, is_cleaned)
                VALUES (?,?,?,?,?,?,?,?,?,?)''', (
                doc['id'], doc.get('title', ''), doc.get('url', ''), doc.get('date', ''),
                doc.get('category', ''), doc.get('source', ''),
                len(doc.get('content', '')),
                len(doc.get('local_images') or doc.get('images') or []),
                doc.get('crawl_time', ''), 1 if doc.get('is_cleaned') else 0,
            ))
            n += 1
        self.conn.commit()
        return n

    def _sync_if_stale(self):
        """JSON 为权威数据源: 若索引文档数与磁盘 *.json 数不一致, 自动重建索引 (自愈)。"""
        try:
            import glob as _glob
            n_files = len(_glob.glob(os.path.join(self.doc_dir, '*.json')))
            if n_files != self.count():
                self.reindex_from_files()
        except Exception:
            pass

    def save(self, doc):
        """保存文档：JSON文件 + SQLite索引"""
        doc_id = doc['id']
        filepath = os.path.join(self.doc_dir, f"{doc_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        self.conn.execute('''INSERT OR REPLACE INTO docs
            (id, title, url, date, category, source, content_length,
             image_count, crawl_time, is_cleaned)
            VALUES (?,?,?,?,?,?,?,?,?,?)''', (
            doc_id, doc.get('title', ''),
            doc.get('url', ''), doc.get('date', ''),
            doc.get('category', ''), doc.get('source', ''),
            len(doc.get('content', '')), len(doc.get('images', [])),
            doc.get('crawl_time', ''), 1 if doc.get('is_cleaned') else 0,
        ))
        self.conn.commit()
        return doc_id

    def load(self, doc_id):
        """加载单篇文档"""
        filepath = os.path.join(self.doc_dir, f"{doc_id}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_all(self, category=None, source=None, date_from=None, date_to=None, limit=None, offset=0):
        """批量加载文档，支持过滤和分页"""
        query = 'SELECT id FROM docs WHERE 1=1'
        params = []
        if category:
            query += ' AND category=?'
            params.append(category)
        if source:
            query += ' AND source=?'
            params.append(source)
        if date_from:
            query += ' AND date >= ?'
            params.append(date_from)
        if date_to:
            query += ' AND date <= ?'
            params.append(date_to)
        query += ' ORDER BY date DESC'
        if limit:
            query += f' LIMIT {limit} OFFSET {offset}'

        docs = []
        for (doc_id,) in self.conn.execute(query, params):
            doc = self.load(doc_id)
            if doc:
                docs.append(doc)
        return docs

    def count(self, **filters):
        """统计文档数量"""
        query = 'SELECT COUNT(*) FROM docs WHERE 1=1'
        params = []
        for k, v in filters.items():
            if v:
                query += f' AND {k}=?'
                params.append(v)
        return self.conn.execute(query, params).fetchone()[0]

    def stats(self):
        """数据统计概览"""
        total = self.count()
        categories = {}
        for (cat, cnt) in self.conn.execute(
            'SELECT category, COUNT(*) FROM docs GROUP BY category'
        ):
            categories[cat] = cnt
        sources = {}
        for (src, cnt) in self.conn.execute(
            'SELECT source, COUNT(*) FROM docs GROUP BY source'
        ):
            sources[src] = cnt
        total_images = self.conn.execute(
            'SELECT SUM(image_count) FROM docs'
        ).fetchone()[0] or 0
        return {
            'total_docs': total,
            'categories': categories,
            'sources': sources,
            'total_images': total_images,
        }

    def close(self):
        self.conn.close()
