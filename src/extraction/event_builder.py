"""
事件组装器
将7个信息点组装成结构化灾害事件
支持JSON序列化、事件卡片生成、时间线排序
"""
from datetime import datetime


class EventBuilder:
    """将信息点组装为结构化事件"""

    # 必需信息点（至少需要类型+地点或时间）
    REQUIRED_FIELDS = {'disaster_type'}
    MIN_FIELDS = 3  # 至少3个非空信息点才构成有效事件

    def build(self, doc, extraction_result):
        """将抽取结果组装为事件"""
        info = extraction_result

        # 计算非空信息点数
        filled = sum(1 for k, v in info.items() if v and (isinstance(v, list) or len(str(v)) > 0))
        filled_fields = [k for k, v in info.items() if v and (isinstance(v, list) or len(str(v)) > 0)]

        # 缺失字段
        missing = [k for k in info if not info[k] or (isinstance(info[k], list) and len(info[k]) == 0)]

        event = {
            'doc_id': doc.get('id', ''),
            'title': doc.get('title', ''),
            'url': doc.get('url', ''),
            'date': doc.get('date', ''),
            'is_valid_event': filled >= self.MIN_FIELDS,
            'filled_count': filled,
            'filled_fields': filled_fields,
            'missing_fields': missing,
            'disaster': {
                'type': info.get('disaster_type', ''),
                'time': info.get('event_time', ''),
                'location': info.get('event_location', ''),
                'casualties': info.get('casualties', ''),
                'economic_loss': info.get('economic_loss', ''),
                'response_level': info.get('response_level', ''),
                'rescue_info': info.get('rescue_info', []),
            },
            # 分类标签
            'severity': self._classify_severity(info),
            'completeness': filled / 7.0,
        }

        # 生成事件摘要
        event['summary'] = self._generate_summary(event)
        return event

    def _classify_severity(self, info):
        """根据信息点判断事件严重程度"""
        text = str(info.get('casualties', '')) + str(info.get('response_level', ''))
        casualties = info.get('casualties', '')

        if 'Ⅰ级' in text or 'I级' in text:
            return '特别重大'
        if 'Ⅱ级' in text or 'II级' in text:
            return '重大'

        # 从伤亡数字判断
        import re
        dead_match = re.search(r'遇难(\d+)人', casualties)
        if dead_match:
            dead = int(dead_match.group(1))
            if dead >= 30:
                return '特别重大'
            if dead >= 10:
                return '重大'
            if dead >= 3:
                return '较大'
            return '一般'

        if 'Ⅲ级' in text or 'III级' in text:
            return '较大'
        if 'Ⅳ级' in text or 'IV级' in text:
            return '一般'

        return '未定级'

    def _generate_summary(self, event):
        """生成事件文字摘要"""
        d = event['disaster']
        parts = []
        if d['time']:
            parts.append(d['time'])
        if d['location']:
            parts.append(d['location'])
        if d['type']:
            parts.append(f'发生{d["type"]}')
        if d['casualties']:
            parts.append(f'造成{d["casualties"]}')
        if d['economic_loss']:
            parts.append(d['economic_loss'])
        if d['response_level']:
            parts.append(f'启动{d["response_level"]}')
        return '，'.join(parts) if parts else '事件信息不完整'

    def build_timeline(self, events):
        """按时间排序生成事件时间线"""
        valid_events = [e for e in events if e['is_valid_event']]
        # 尝试按时间排序
        def sort_key(e):
            t = e['disaster'].get('time', '')
            # 尝试解析日期
            import re
            m = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', t)
            if m:
                return f'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}'
            m2 = re.search(r'(\d{4}-\d{2}-\d{2})', t)
            if m2:
                return m2.group(1)
            return '9999-99-99'
        return sorted(valid_events, key=sort_key)

    def build_geo_data(self, events):
        """生成地理分布数据（用于地图可视化）"""
        geo = {}
        for e in events:
            loc = e['disaster'].get('location', '')
            if not loc:
                continue
            if loc not in geo:
                geo[loc] = []
            geo[loc].append({
                'type': e['disaster']['type'],
                'time': e['disaster']['time'],
                'severity': e['severity'],
                'doc_id': e['doc_id'],
            })
        return geo
