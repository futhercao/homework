"""
知识图谱构建与可视化模块（创新点）
从抽取的结构化灾害事件中构建事件知识图谱：
- 节点：事件、地点、组织、灾害类型、时间
- 边：发生于、参与救援、造成、响应级别 等关系
- 可视化：生成 vis.js 交互式网络图
"""
import json
import os
from collections import defaultdict

from config import KG_DIR


class EventKnowledgeGraph:
    """事件知识图谱"""

    NODE_TYPES = {
        'event': {'color': '#e74c3c', 'shape': 'star', 'label': '事件'},
        'location': {'color': '#3498db', 'shape': 'dot', 'label': '地点'},
        'disaster_type': {'color': '#e67e22', 'shape': 'diamond', 'label': '灾害类型'},
        'organization': {'color': '#2ecc71', 'shape': 'triangle', 'label': '组织'},
        'time': {'color': '#9b59b6', 'shape': 'square', 'label': '时间'},
        'response_level': {'color': '#f1c40f', 'shape': 'triangleDown', 'label': '响应级别'},
    }

    EDGE_TYPES = {
        'occurred_at': {'label': '发生于', 'color': '#3498db'},
        'is_type': {'label': '属于', 'color': '#e67e22'},
        'rescued_by': {'label': '救援', 'color': '#2ecc71'},
        'occurred_on': {'label': '发生时间', 'color': '#9b59b6'},
        'response': {'label': '响应级别', 'color': '#f1c40f'},
        'caused': {'label': '造成', 'color': '#e74c3c'},
    }

    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.events = []

    def add_event(self, doc_id, extraction_result):
        """从抽取结果添加事件到知识图谱"""
        event = extraction_result.get('event', extraction_result.get('fused', {}))
        if not event:
            return

        event_id = f'event_{doc_id}'
        dtype = event.get('disaster_type', '未知')
        location = event.get('event_location', '未知')
        title = f'{location}{dtype}'

        self._add_node(event_id, title, 'event', extra={
            'casualties': event.get('casualties', ''),
            'economic_loss': event.get('economic_loss', ''),
        })

        if dtype:
            type_id = f'type_{dtype}'
            self._add_node(type_id, dtype, 'disaster_type')
            self._add_edge(event_id, type_id, 'is_type')

        if location and location != '未知':
            loc_id = f'loc_{location}'
            self._add_node(loc_id, location, 'location')
            self._add_edge(event_id, loc_id, 'occurred_at')

        time_str = event.get('event_time', '')
        if time_str:
            time_id = f'time_{time_str}'
            self._add_node(time_id, time_str, 'time')
            self._add_edge(event_id, time_id, 'occurred_on')

        level = event.get('response_level', '')
        if level:
            level_id = f'level_{level}'
            self._add_node(level_id, f'{level}响应', 'response_level')
            self._add_edge(event_id, level_id, 'response')

        orgs = event.get('rescue_org', [])
        if isinstance(orgs, str):
            orgs = [orgs] if orgs else []
        for org in orgs[:3]:
            org_id = f'org_{org}'
            self._add_node(org_id, org, 'organization')
            self._add_edge(event_id, org_id, 'rescued_by')

        self.events.append({
            'doc_id': doc_id, 'event_id': event_id, 'event': event,
        })

    def _add_node(self, node_id, label, node_type, extra=None):
        if node_id not in self.nodes:
            type_config = self.NODE_TYPES.get(node_type, {})
            self.nodes[node_id] = {
                'id': node_id,
                'label': label[:20],
                'title': label,
                'type': node_type,
                'color': type_config.get('color', '#999'),
                'shape': type_config.get('shape', 'dot'),
                'size': 25 if node_type == 'event' else 15,
            }
            if extra:
                self.nodes[node_id].update(extra)

    def _add_edge(self, from_id, to_id, edge_type):
        edge_config = self.EDGE_TYPES.get(edge_type, {})
        self.edges.append({
            'from': from_id,
            'to': to_id,
            'label': edge_config.get('label', ''),
            'color': {'color': edge_config.get('color', '#999')},
            'arrows': 'to',
        })

    def to_vis_data(self):
        """导出 vis.js 可视化数据"""
        return {
            'nodes': list(self.nodes.values()),
            'edges': self.edges,
        }

    def get_statistics(self):
        """获取知识图谱统计信息"""
        type_counts = defaultdict(int)
        for node in self.nodes.values():
            type_counts[node['type']] += 1

        location_events = defaultdict(list)
        type_events = defaultdict(list)
        for event in self.events:
            loc = event['event'].get('event_location', '未知')
            dtype = event['event'].get('disaster_type', '未知')
            location_events[loc].append(event['doc_id'])
            type_events[dtype].append(event['doc_id'])

        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'total_events': len(self.events),
            'node_type_counts': dict(type_counts),
            'top_locations': dict(
                sorted(location_events.items(), key=lambda x: -len(x[1]))[:10]
            ),
            'disaster_type_distribution': {k: len(v) for k, v in type_events.items()},
        }

    def save(self, path=None):
        """保存知识图谱数据"""
        os.makedirs(KG_DIR, exist_ok=True)
        path = path or os.path.join(KG_DIR, 'knowledge_graph.json')
        data = {
            'nodes': list(self.nodes.values()),
            'edges': self.edges,
            'events': self.events,
            'statistics': self.get_statistics(),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[KG] 知识图谱已保存: {len(self.nodes)} 个节点, {len(self.edges)} 条边")

    def load(self, path=None):
        """加载知识图谱数据"""
        path = path or os.path.join(KG_DIR, 'knowledge_graph.json')
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.nodes = {n['id']: n for n in data.get('nodes', [])}
        self.edges = data.get('edges', [])
        self.events = data.get('events', [])
        return True
