"""
数据采集模块
- DisasterDataGenerator: 生成高仿真自然灾害新闻语料（含多媒体）
- WebCrawler: 从新闻网站爬取真实灾害新闻（可选）
- 为每篇文档生成信息图（模拟新闻配图），用于多媒体信息抽取
"""
import os
import json
import random
import time
import uuid
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import DATA_DIR, IMAGE_DIR, CRAWLER_CONFIG

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ──────────────────────────────────────────────────────────────
# 灾害新闻样本数据生成器
# ──────────────────────────────────────────────────────────────

class DisasterDataGenerator:
    """
    生成高仿真自然灾害新闻文档，涵盖7类灾害，
    每篇文章自然嵌入7个信息点（灾害类型、时间、地点、伤亡、
    经济损失、响应级别、救援组织），用于信息抽取实验。
    同时为部分文章生成信息图图片，支持多媒体信息抽取。
    """

    DISASTER_TYPES = ['地震', '台风', '洪水', '山体滑坡', '火灾', '干旱', '暴雪']

    PROVINCES = [
        '四川省', '云南省', '甘肃省', '青海省', '西藏自治区',
        '广东省', '福建省', '浙江省', '江苏省', '山东省',
        '河南省', '湖北省', '湖南省', '安徽省', '江西省',
        '贵州省', '广西壮族自治区', '重庆市', '陕西省', '河北省',
        '辽宁省', '吉林省', '黑龙江省', '内蒙古自治区', '新疆维吾尔自治区',
    ]

    CITIES = {
        '四川省': ['成都市', '绵阳市', '德阳市', '广元市', '雅安市', '宜宾市', '泸州市', '阿坝州'],
        '云南省': ['昆明市', '大理市', '丽江市', '昭通市', '曲靖市', '玉溪市'],
        '甘肃省': ['兰州市', '天水市', '陇南市', '定西市', '临夏州'],
        '广东省': ['广州市', '深圳市', '珠海市', '汕头市', '湛江市', '茂名市'],
        '福建省': ['福州市', '厦门市', '泉州市', '漳州市', '宁德市'],
        '浙江省': ['杭州市', '宁波市', '温州市', '台州市', '丽水市'],
        '江苏省': ['南京市', '苏州市', '无锡市', '常州市', '盐城市'],
        '山东省': ['济南市', '青岛市', '烟台市', '潍坊市', '临沂市'],
        '河南省': ['郑州市', '洛阳市', '开封市', '新乡市', '信阳市', '南阳市'],
        '湖北省': ['武汉市', '宜昌市', '荆州市', '十堰市', '恩施州'],
        '湖南省': ['长沙市', '岳阳市', '常德市', '湘西州', '衡阳市'],
        '安徽省': ['合肥市', '芜湖市', '安庆市', '六安市', '黄山市'],
        '贵州省': ['贵阳市', '遵义市', '毕节市', '六盘水市', '铜仁市'],
        '重庆市': ['万州区', '涪陵区', '黔江区', '渝北区', '巫山县'],
        '陕西省': ['西安市', '宝鸡市', '汉中市', '安康市', '商洛市'],
        '河北省': ['石家庄市', '唐山市', '邯郸市', '保定市', '张家口市'],
        '辽宁省': ['沈阳市', '大连市', '鞍山市', '丹东市', '锦州市'],
        '吉林省': ['长春市', '吉林市', '延边州', '通化市'],
        '黑龙江省': ['哈尔滨市', '齐齐哈尔市', '牡丹江市', '大庆市'],
    }

    RESPONSE_LEVELS = ['I级', 'II级', 'III级', 'IV级']

    ORGS = [
        '国家应急管理部', '中国红十字会', '国家消防救援局',
        '解放军某部', '武警部队', '蓝天救援队', '中国地震局',
        '国家气象局', '民政部', '当地消防支队', '省应急管理厅',
        '国家防汛抗旱总指挥部', '中国气象局', '中国国际救援队',
        '公益救援联盟', '绿舟救援队', '国家减灾委员会',
    ]

    # 灾害类型→不适用组织的排除规则（避免火灾由防汛指挥部处理等语义错误）
    ORG_EXCLUDE = {
        '地震': [],  # 地震时所有组织都可能参与
        '台风': [],
        '洪水': [],
        '山体滑坡': [],
        '火灾': ['国家防汛抗旱总指挥部', '国家防总', '水利专家'],
        '干旱': ['国家防汛抗旱总指挥部', '国家防总'],
        '暴雪': [],
    }

    DISASTER_SPECIFIC_ORGS = {
        '地震': ['中国地震局'],
        '台风': ['国家气象局', '中国气象局'],
        '洪水': ['国家防汛抗旱总指挥部', '水利专家'],
        '干旱': ['水利专家'],
    }

    TEMPLATES = {
        '地震': [
            (
                '{province}{city}发生{mag}级地震 已造成{dead}人遇难',
                '{date}，{province}{city}发生里氏{mag}级地震，震源深度{depth}公里。'
                '据{org1}消息，截至目前，地震已造成{dead}人遇难、{injured}人受伤，'
                '紧急转移安置{evacuated}万人。初步统计，地震造成直接经济损失约{loss}亿元。'
                '{province}已启动{level}应急响应，{org2}紧急调拨救灾物资并派出救援队伍赶赴灾区。'
                '目前，搜救工作仍在紧张进行中，余震活动频繁，最大余震达{mag2}级。'
                '专家指出，此次地震位于{fault}断裂带附近，震区建筑抗震设防烈度需要重新评估。'
                '国务院已派出工作组指导救灾工作，要求全力搜救被困人员，妥善安置受灾群众。'
            ),
            (
                '{province}{city}地震灾区救援持续进行',
                '{date}，{province}{city}发生{mag}级地震后，各方救援力量迅速集结。'
                '{org1}已派出{teams}支救援队伍、{people}余名救援人员赶赴灾区。'
                '经过连续{hours}小时的搜救，已从废墟中成功营救出{rescued}名被困群众。'
                '截至目前，地震共造成{dead}人遇难、{injured}人受伤，直接经济损失{loss}亿元。'
                '{province}政府宣布启动{level}应急响应，{org2}紧急下拨{fund}万元救灾资金。'
                '医疗救治、受灾群众安置、基础设施抢修等各项工作正有序推进。'
                '气象部门预报灾区未来几天将有降雨，相关部门加强地质灾害防范工作。'
            ),
        ],
        '台风': [
            (
                '台风"{typhoon_name}"在{province}登陆 最大风力{wind}级',
                '{date}，今年第{typhoon_no}号台风"{typhoon_name}"在{province}{city}沿海登陆，'
                '登陆时中心附近最大风力{wind}级（{wind_speed}米/秒），中心最低气压{pressure}百帕。'
                '受台风影响，{province}沿海地区出现暴雨到大暴雨，局部特大暴雨，'
                '最大累计降雨量达{rain}毫米。台风造成{dead}人遇难、{missing}人失踪、'
                '{injured}人受伤，紧急转移安置{evacuated}万人，直接经济损失约{loss}亿元。'
                '{org1}已启动{level}应急响应，{org2}提前部署防台风工作。'
                '目前台风已减弱为热带低压，但后续降雨仍需警惕。'
            ),
            (
                '超强台风"{typhoon_name}"致{province}多地受灾严重',
                '{date}，超强台风"{typhoon_name}"（今年第{typhoon_no}号台风）对{province}'
                '造成严重影响。{city}等地出现{wind}级大风，沿海浪高达{wave}米。'
                '暴雨引发山洪、泥石流等次生灾害，造成{dead}人遇难、{injured}人受伤。'
                '全省共有{counties}个县（市、区）受灾，受灾人口{affected}万人，'
                '直接经济损失达{loss}亿元。{province}已启动{level}应急响应，'
                '{org1}调集{people}名消防指战员参与救援，{org2}紧急调拨帐篷、'
                '棉被等救灾物资{supplies}万件。国家防总要求做好灾后恢复重建工作。'
            ),
        ],
        '洪水': [
            (
                '{province}{city}遭遇特大洪水 {river}水位超警戒',
                '{date}，受持续强降雨影响，{province}{city}境内{river}水位急剧上涨，'
                '超过警戒水位{over_level}米，达到{flood_level}米，为近{years}年来最高水位。'
                '洪水导致{dead}人遇难、{missing}人失踪，{houses}间房屋倒塌或严重受损，'
                '农作物受灾面积达{crop}万亩，直接经济损失约{loss}亿元。'
                '{province}紧急启动{level}防汛应急响应，{org1}派出工作组赶赴现场指导抗洪抢险。'
                '{org2}出动{soldiers}名官兵协助转移群众{evacuated}万人。'
                '水利专家指出，上游水库已开始调蓄洪水，下游河道行洪能力需进一步提升。'
            ),
            (
                '{province}多地暴发洪涝灾害 {river}干流全线超警',
                '{date}，{province}遭遇入汛以来最强降雨过程，{city}等地24小时降雨量超过{rain}毫米。'
                '{river}干流全线超过警戒水位，{river2}流域也出现超保证水位。'
                '截至目前，洪涝灾害已造成{dead}人遇难、{injured}人受伤，'
                '紧急转移安置{evacuated}万人，直接经济损失{loss}亿元。'
                '{org1}启动{level}应急响应，{org2}紧急调拨{supplies}万元救灾资金。'
                '国家防汛抗旱总指挥部要求切实做好水库安全度汛、'
                '中小河流洪水防御和山洪地质灾害防范工作。'
            ),
        ],
        '山体滑坡': [
            (
                '{province}{city}发生山体滑坡 {buried}人被埋',
                '{date}，{province}{city}{county}因连日暴雨引发大面积山体滑坡，'
                '约{volume}万立方米土石方倾泻而下，造成{buried}人被埋、{houses}间房屋被掩埋。'
                '事故发生后，{org1}立即启动{level}应急响应，{org2}紧急调派{teams}支'
                '救援队伍、{people}名救援人员及{equipment}台大型机械赶赴现场。'
                '经过{hours}小时的连续搜救，已找到{found}名被埋人员，其中{dead}人遇难、'
                '{rescued}人获救。初步估算直接经济损失约{loss}亿元。'
                '专家分析指出，当地地质条件复杂，加上前期降雨量大，土壤含水量饱和是'
                '引发此次山体滑坡的主要原因。相关部门已对周边区域开展地质灾害排查。'
            ),
        ],
        '火灾': [
            (
                '{province}{city}发生{fire_type}火灾 过火面积{area}公顷',
                '{date}，{province}{city}{county}发生{fire_type}火灾。'
                '火灾发生后，{org1}紧急启动{level}应急响应，调派{people}名消防指战员、'
                '{planes}架直升机投入灭火作战。经过{hours}小时的全力扑救，'
                '明火已于{ext_date}全部扑灭。火灾共造成过火面积约{area}公顷，'
                '{dead}人遇难、{injured}人受伤，直接经济损失约{loss}亿元。'
                '{org2}已派出专家组调查火灾原因。初步调查显示，{cause}是引发此次火灾的原因。'
                '当地政府要求各地深刻吸取教训，加强防火巡护和火源管控。'
            ),
        ],
        '干旱': [
            (
                '{province}遭遇严重干旱 {affected}万人饮水困难',
                '{date}以来，{province}{city}等地持续高温少雨，遭遇近{years}年来最严重干旱。'
                '据{org1}统计，全省共有{counties}个县（市、区）受灾，'
                '受灾人口达{affected}万人，其中{drink}万人出现饮水困难。'
                '农作物受旱面积达{crop}万亩，其中绝收{dead_crop}万亩，'
                '直接经济损失约{loss}亿元。{province}已启动{level}抗旱应急响应，'
                '{org2}紧急调拨{fund}万元抗旱救灾资金，组织送水车{trucks}辆保障群众饮水。'
                '气象部门预计，旱情仍将持续，建议各地加强水资源调度和节水措施。'
            ),
        ],
        '暴雪': [
            (
                '{province}遭受暴雪袭击 多条高速公路封闭',
                '{date}，{province}{city}等地迎来今冬最强暴雪天气，'
                '最大积雪深度达{snow}厘米，局部地区出现暴风雪。'
                '暴雪导致{roads}条高速公路封闭、{flights}个航班取消，'
                '{trains}趟列车晚点或停运。截至目前，暴雪已造成{dead}人遇难、'
                '{injured}人受伤，{houses}间房屋因积雪过重倒塌，'
                '直接经济损失约{loss}亿元。{org1}启动{level}应急响应，'
                '{org2}出动除雪机械{machines}台次，清除积雪{clear}万立方米。'
                '各地民政部门紧急发放御寒物资，确保受灾群众温暖过冬。'
            ),
        ],
    }

    TYPHOON_NAMES = [
        '海葵', '杜苏芮', '苏拉', '卡努', '泰利', '梅花', '暹芭',
        '奥鹿', '玫瑰', '桑达', '翠丝', '万宜', '天兔', '帕布',
        '蝴蝶', '圣帕', '韦帕', '丹娜丝', '百合', '玲玲', '剑鱼',
    ]

    RIVERS = ['长江', '黄河', '珠江', '淮河', '松花江', '海河', '嘉陵江',
              '汉江', '湘江', '赣江', '闽江', '钱塘江', '沅江', '澧水']

    FAULTS = ['龙门山', '鲜水河', '小江', '红河', '海原', '郯庐',
              '汾渭', '天山北缘', '阿尔金', '昆仑山']

    FIRE_TYPES = ['森林', '草原', '山林']
    FIRE_CAUSES = [
        '人为用火不慎', '雷击引燃枯草', '电线短路', '祭祀用火失控',
        '农事用火引发', '自燃起火',
    ]

    COUNTIES = [
        '北川县', '汶川县', '茂县', '理县', '映秀镇', '都江堰市',
        '彝良县', '鲁甸县', '巧家县', '威宁县', '赫章县',
        '舟曲县', '文县', '武都区', '康县', '成县',
    ]

    @classmethod
    def generate(cls, num_docs=150, save=True):
        """生成灾害新闻样本文档"""
        documents = []
        random.seed(42)
        base_date = datetime(2024, 1, 1)

        while len(documents) < num_docs:
            for dtype in cls.DISASTER_TYPES:
                if len(documents) >= num_docs:
                    break
                templates = cls.TEMPLATES[dtype]
                for tmpl_title, tmpl_content in templates:
                    if len(documents) >= num_docs:
                        break

                    for _ in range(random.randint(2, 4)):
                        if len(documents) >= num_docs:
                            break

                        doc_id = f'doc_{len(documents):04d}'
                        province = random.choice(cls.PROVINCES)
                        cities = cls.CITIES.get(province, ['某市'])
                        city = random.choice(cities)
                        date_obj = base_date + timedelta(days=random.randint(0, 700))
                        date_str = date_obj.strftime('%Y年%m月%d日')
                        level = random.choice(cls.RESPONSE_LEVELS)
                        # 根据灾害类型排除不适用的组织
                        exclude = cls.ORG_EXCLUDE.get(dtype, [])
                        allowed_orgs = [o for o in cls.ORGS if o not in exclude]
                        # 优先选择灾害特定组织
                        specific = cls.DISASTER_SPECIFIC_ORGS.get(dtype, [])
                        if specific and random.random() < 0.5:
                            org1 = random.choice(specific)
                        else:
                            org1 = random.choice(allowed_orgs)
                        org2 = random.choice([o for o in allowed_orgs if o != org1])

                        ground_truth = {
                            'disaster_type': dtype,
                            'event_time': date_str,
                            'event_location': f'{province}{city}',
                            'response_level': level,
                            'rescue_org': [org1, org2],
                        }

                        params = cls._make_params(
                            dtype, province, city, date_str, date_obj, level, org1, org2
                        )
                        ground_truth['casualties'] = params.get('_casualties', '')
                        ground_truth['economic_loss'] = params.get('_loss', '')

                        try:
                            title = tmpl_title.format(**params)
                            content = tmpl_content.format(**params)
                        except (KeyError, IndexError):
                            continue

                        image_info = None
                        if HAS_PIL and random.random() < 0.6:
                            image_info = cls._generate_infographic(
                                doc_id, dtype, ground_truth
                            )

                        doc = {
                            'id': doc_id,
                            'title': title,
                            'content': content,
                            'url': f'https://disaster-news.example.com/article/{doc_id}',
                            'date': date_obj.strftime('%Y-%m-%d'),
                            'disaster_type': dtype,
                            'images': [image_info] if image_info else [],
                            'ground_truth': ground_truth,
                            'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        documents.append(doc)

        if save:
            cls._save(documents)

        return documents

    @classmethod
    def _make_params(cls, dtype, province, city, date_str, date_obj, level, org1, org2):
        """根据灾害类型生成参数"""
        dead = random.randint(0, 85)
        injured = random.randint(dead, dead + 200)
        loss = round(random.uniform(0.5, 120), 1)

        params = {
            'province': province, 'city': city, 'date': date_str,
            'level': level, 'org1': org1, 'org2': org2,
            'dead': dead, 'injured': injured,
            'loss': loss,
            'evacuated': round(random.uniform(0.5, 30), 1),
            '_casualties': f'遇难{dead}人，受伤{injured}人',
            '_loss': f'{loss}亿元',
        }

        if dtype == '地震':
            params.update({
                'mag': round(random.uniform(4.5, 8.0), 1),
                'mag2': round(random.uniform(3.0, 5.5), 1),
                'depth': random.randint(5, 30),
                'fault': random.choice(cls.FAULTS),
                'teams': random.randint(10, 50),
                'people': random.randint(500, 5000),
                'hours': random.randint(12, 72),
                'rescued': random.randint(5, 50),
                'fund': random.randint(500, 5000),
            })
        elif dtype == '台风':
            params.update({
                'typhoon_name': random.choice(cls.TYPHOON_NAMES),
                'typhoon_no': random.randint(1, 25),
                'wind': random.randint(10, 17),
                'wind_speed': random.randint(25, 62),
                'pressure': random.randint(920, 980),
                'rain': random.randint(100, 600),
                'missing': random.randint(0, 10),
                'wave': round(random.uniform(3, 14), 1),
                'counties': random.randint(5, 30),
                'affected': round(random.uniform(10, 200), 1),
                'people': random.randint(500, 5000),
                'supplies': round(random.uniform(1, 50), 1),
            })
        elif dtype == '洪水':
            params.update({
                'river': random.choice(cls.RIVERS),
                'river2': random.choice(cls.RIVERS),
                'over_level': round(random.uniform(0.5, 5), 1),
                'flood_level': round(random.uniform(20, 40), 1),
                'years': random.randint(20, 100),
                'missing': random.randint(0, 15),
                'houses': random.randint(100, 5000),
                'crop': round(random.uniform(5, 100), 1),
                'soldiers': random.randint(500, 5000),
                'rain': random.randint(100, 400),
                'counties': random.randint(5, 20),
                'supplies': random.randint(500, 5000),
            })
        elif dtype == '山体滑坡':
            buried = random.randint(5, 50)
            found = random.randint(buried - 3, buried)
            rescued = max(0, found - dead)
            params.update({
                'county': random.choice(cls.COUNTIES),
                'volume': round(random.uniform(5, 100), 0),
                'buried': buried,
                'houses': random.randint(5, 30),
                'teams': random.randint(5, 20),
                'people': random.randint(200, 2000),
                'equipment': random.randint(10, 50),
                'hours': random.randint(12, 96),
                'found': found,
                'rescued': rescued,
            })
        elif dtype == '火灾':
            params.update({
                'county': random.choice(cls.COUNTIES) if random.random() < 0.3
                    else f'{city}某区',
                'fire_type': random.choice(cls.FIRE_TYPES),
                'area': random.randint(50, 2000),
                'planes': random.randint(2, 10),
                'people': random.randint(500, 5000),
                'hours': random.randint(6, 72),
                'ext_date': (date_obj + timedelta(days=1)).strftime('%Y年%m月%d日'),
                'cause': random.choice(cls.FIRE_CAUSES),
            })
        elif dtype == '干旱':
            params.update({
                'years': random.randint(20, 60),
                'counties': random.randint(10, 50),
                'affected': round(random.uniform(50, 500), 1),
                'drink': round(random.uniform(5, 80), 1),
                'crop': round(random.uniform(50, 500), 1),
                'dead_crop': round(random.uniform(5, 50), 1),
                'fund': random.randint(1000, 10000),
                'trucks': random.randint(50, 500),
            })
            params['_casualties'] = f'受灾{params["affected"]}万人'
        elif dtype == '暴雪':
            params.update({
                'snow': random.randint(20, 80),
                'roads': random.randint(3, 15),
                'flights': random.randint(50, 300),
                'trains': random.randint(10, 100),
                'houses': random.randint(10, 200),
                'machines': random.randint(100, 1000),
                'clear': round(random.uniform(5, 50), 1),
            })

        return params

    @classmethod
    def _generate_infographic(cls, doc_id, dtype, gt):
        """用 Pillow 生成灾害信息图图片"""
        os.makedirs(IMAGE_DIR, exist_ok=True)
        W, H = 600, 400

        bg_colors = {
            '地震': (180, 60, 60), '台风': (50, 100, 160), '洪水': (40, 90, 150),
            '山体滑坡': (120, 100, 60), '火灾': (200, 80, 30),
            '干旱': (190, 160, 80), '暴雪': (160, 180, 200),
        }
        bg = bg_colors.get(dtype, (100, 100, 100))

        img = Image.new('RGB', (W, H), bg)
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("msyh.ttc", 28)
            font_body = ImageFont.truetype("msyh.ttc", 18)
            font_small = ImageFont.truetype("msyh.ttc", 14)
        except (OSError, IOError):
            try:
                font_title = ImageFont.truetype("simhei.ttf", 28)
                font_body = ImageFont.truetype("simhei.ttf", 18)
                font_small = ImageFont.truetype("simhei.ttf", 14)
            except (OSError, IOError):
                font_title = ImageFont.load_default()
                font_body = font_title
                font_small = font_title

        draw.rectangle([0, 0, W, 60], fill=(0, 0, 0, 128))
        draw.text((20, 12), f"灾害速报 | {dtype}", fill='white', font=font_title)

        y = 80
        info_lines = [
            f"灾害类型：{gt['disaster_type']}",
            f"发生时间：{gt['event_time']}",
            f"发生地点：{gt['event_location']}",
            f"伤亡情况：{gt['casualties']}",
            f"经济损失：{gt['economic_loss']}",
            f"响应级别：{gt['response_level']}",
        ]
        for line in info_lines:
            draw.text((30, y), line, fill='white', font=font_body)
            y += 35

        draw.text((30, H - 35), "数据来源：国家应急管理部", fill=(200, 200, 200), font=font_small)

        draw.rectangle([W - 130, 70, W - 20, 170], outline='white', width=2)
        draw.text((W - 120, 80), dtype, fill='white', font=font_title)
        draw.text((W - 120, 120), gt['response_level'], fill=(255, 200, 0), font=font_body)

        img_path = os.path.join(IMAGE_DIR, f'{doc_id}.png')
        img.save(img_path, 'PNG')

        ocr_text = '\n'.join(info_lines)
        return {
            'path': img_path,
            'filename': f'{doc_id}.png',
            'alt': f'{dtype}灾害信息图',
            'ocr_text': ocr_text,
        }

    @classmethod
    def _save(cls, documents):
        """保存文档到本地"""
        os.makedirs(DATA_DIR, exist_ok=True)
        for doc in documents:
            path = os.path.join(DATA_DIR, f"{doc['id']}.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

        manifest = {
            'total': len(documents),
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': 'disaster_generator',
            'disaster_stats': {},
            'docs': [{'id': d['id'], 'title': d['title'], 'type': d['disaster_type']}
                     for d in documents],
        }
        for d in documents:
            t = d['disaster_type']
            manifest['disaster_stats'][t] = manifest['disaster_stats'].get(t, 0) + 1

        with open(os.path.join(DATA_DIR, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[生成器] 已生成 {len(documents)} 篇灾害新闻到 {DATA_DIR}")


# ──────────────────────────────────────────────────────────────
# 通用网页爬虫
# ──────────────────────────────────────────────────────────────

class WebCrawler:
    """从新闻网站爬取灾害新闻"""

    def __init__(self, config=None):
        self.config = config or CRAWLER_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config['user_agent'],
            'Accept-Language': 'zh-CN,zh;q=0.9',
        })
        self.documents = []
        self.visited = set()

    def crawl(self, seed_urls=None, max_docs=None):
        seed_urls = seed_urls or self.config['seed_urls']
        max_docs = max_docs or self.config['max_documents']
        queue = list(seed_urls)

        print(f"[爬虫] 开始爬取灾害新闻，目标: {max_docs} 篇")

        while queue and len(self.documents) < max_docs:
            url = queue.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)
            try:
                resp = self.session.get(url, timeout=self.config.get('request_timeout', 15))
                resp.encoding = resp.apparent_encoding or 'utf-8'
                if resp.status_code != 200:
                    continue
                html = resp.text
                soup = BeautifulSoup(html, 'lxml')
                title_el = soup.find('h1') or soup.find('title')
                title = title_el.get_text(strip=True) if title_el else ''
                content_el = soup.select_one('article') or soup.select_one('.article-content')
                if content_el:
                    content = content_el.get_text(separator='\n', strip=True)
                else:
                    content = ''

                if len(content) >= 80:
                    doc = {
                        'id': f'crawl_{len(self.documents):04d}',
                        'title': title, 'content': content,
                        'url': url,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'images': [],
                        'crawl_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                    self.documents.append(doc)
                    print(f"  [{len(self.documents)}/{max_docs}] {title[:40]}")

                for a in soup.find_all('a', href=True)[:30]:
                    href = a['href']
                    if not href.startswith('http'):
                        href = urljoin(url, href)
                    if urlparse(href).netloc == urlparse(url).netloc:
                        queue.append(href)

                time.sleep(self.config.get('request_delay', 1))
            except Exception as e:
                print(f"  [错误] {url}: {e}")

        if self.documents:
            os.makedirs(DATA_DIR, exist_ok=True)
            for doc in self.documents:
                path = os.path.join(DATA_DIR, f"{doc['id']}.json")
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"[爬虫] 完成，共获取 {len(self.documents)} 篇文档")
        return self.documents


def load_documents():
    """加载本地存储的所有文档"""
    documents = []
    if not os.path.exists(DATA_DIR):
        return documents
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.endswith('.json') and fname != 'manifest.json':
            with open(os.path.join(DATA_DIR, fname), 'r', encoding='utf-8') as f:
                documents.append(json.load(f))
    return documents


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--crawl':
        crawler = WebCrawler()
        crawler.crawl()
    else:
        DisasterDataGenerator.generate(150)
