"""
知识图谱构建器
基于抽取的灾害事件信息构建知识图谱
- 实体: 灾害类型、地点、时间、组织
- 关系: 发生地点、造成伤亡、经济损失、响应级别、救援行动
- 输出: JSON格式图数据（可用于ECharts/D3.js可视化）
"""
import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import DATA_DIR


class KnowledgeGraphBuilder:
    """灾害事件知识图谱构建"""

    def __init__(self):
        self.entities = {}       # {entity_id: {type, name, ...}}
        self.relations = []      # [{source, target, relation, ...}]
        self.entity_counters = defaultdict(int)

    def build(self, events):
        """从事件列表构建知识图谱"""
        for event in events:
            if not event['is_valid_event']:
                continue

            d = event['disaster']
            doc_entity = self._add_entity('DOCUMENT', event.get('title', '')[:50],
                                           {'doc_id': event['doc_id']})

            # 灾害类型实体
            if d['type']:
                type_entity = self._add_entity('DISASTER_TYPE', d['type'])
                self._add_relation(doc_entity, type_entity, 'HAS_TYPE')

            # 地点实体
            if d['location']:
                loc_entity = self._add_entity('LOCATION', d['location'])
                self._add_relation(doc_entity, loc_entity, 'OCCURRED_AT')

            # 时间实体
            if d['time']:
                time_entity = self._add_entity('TIME', d['time'])
                self._add_relation(doc_entity, time_entity, 'OCCURRED_ON')

            # 伤亡实体
            if d['casualties']:
                cas_entity = self._add_entity('CASUALTIES', d['casualties'])
                self._add_relation(doc_entity, cas_entity, 'CAUSED')

            # 经济损失实体
            if d['economic_loss']:
                econ_entity = self._add_entity('ECONOMIC_LOSS', d['economic_loss'])
                self._add_relation(doc_entity, econ_entity, 'LOST')

            # 响应级别
            if d['response_level']:
                resp_entity = self._add_entity('RESPONSE', d['response_level'])
                self._add_relation(doc_entity, resp_entity, 'ACTIVATED')

            # 救援组织
            for org in d.get('rescue_info', []):
                if isinstance(org, str) and len(org) >= 3:
                    org_entity = self._add_entity('ORGANIZATION', org)
                    self._add_relation(org_entity, doc_entity, 'RESCUED')

            # 严重程度
            severity = event.get('severity', '未定级')
            sev_entity = self._add_entity('SEVERITY', severity)
            self._add_relation(doc_entity, sev_entity, 'HAS_SEVERITY')

        return self.export()

    def _add_entity(self, etype, name, extra=None):
        eid = f"{etype}_{name}"
        if eid not in self.entities:
            self.entities[eid] = {
                'id': eid,
                'type': etype,
                'name': name,
                'category': self._get_category(etype),
                **(extra or {}),
            }
        return eid

    def _add_relation(self, source, target, relation):
        if source in self.entities and target in self.entities:
            # 避免重复关系
            key = (source, target, relation)
            if not any(r['source'] == source and r['target'] == target
                      and r['relation'] == relation for r in self.relations):
                self.relations.append({
                    'source': source,
                    'target': target,
                    'relation': relation,
                })

    def _get_category(self, etype):
        categories = {
            'DOCUMENT': 0, 'DISASTER_TYPE': 1, 'LOCATION': 2, 'TIME': 3,
            'CASUALTIES': 4, 'ECONOMIC_LOSS': 5, 'RESPONSE': 6,
            'ORGANIZATION': 7, 'SEVERITY': 8,
        }
        return categories.get(etype, 9)

    def export(self):
        """导出为可视化友好的格式"""
        nodes = []
        for eid, e in self.entities.items():
            nodes.append({
                'id': eid,
                'name': e['name'],
                'type': e['type'],
                'category': e['category'],
                'symbolSize': self._get_node_size(e['type']),
            })

        links = []
        for r in self.relations:
            links.append({
                'source': r['source'],
                'target': r['target'],
                'label': r['relation'],
            })

        return {
            'nodes': nodes,
            'links': links,
            'stats': {
                'total_nodes': len(nodes),
                'total_links': len(links),
                'total_events': sum(1 for n in nodes if n['type'] == 'DOCUMENT'),
            },
        }

    def _get_node_size(self, etype):
        sizes = {'DOCUMENT': 20, 'DISASTER_TYPE': 30, 'LOCATION': 25,
                  'TIME': 15, 'CASUALTIES': 20, 'ECONOMIC_LOSS': 20,
                  'RESPONSE': 20, 'ORGANIZATION': 18, 'SEVERITY': 25}
        return sizes.get(etype, 15)

    def save(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(DATA_DIR, 'knowledge_graph.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export(), f, ensure_ascii=False, indent=2)
        print(f"[知识图谱] 已保存: {len(self.entities)} 实体, {len(self.relations)} 关系")
        return filepath
