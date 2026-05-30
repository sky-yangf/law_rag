"""
LAW_RAG 核心检索模块
- 关键词召回：解决向量相似度不足的法条命中（如帮信罪、信用卡诈骗等）
- 双集合检索：statutes(微调bge) + cail_cases(bge-large)
- RRF 融合：双集合结果按 reciprocal rank 融合
"""
import re
from typing import List
from langchain_core.documents import Document

# ============================================================
# 0. 关键词召回表
# ============================================================
STATUTE_KEYWORDS = [
    {
        'keywords': ['帮信罪', '帮助信息网络犯罪', '出借银行卡', '借银行卡', '租借银行卡',
                     '出借账户', '借微信', '借支付宝', '租借账户', '账户出借'],
        'pk': '中华人民共和国刑法_第二百八十七条之一',
        'law_name': '中华人民共和国刑法',
        'article': '第二百八十七条之一',
        'text': '利用信息网络实施下列行为之一，情节严重的，处三年以下有期徒刑或者拘役，并处或者单处罚金：（一）设立用于实施诈骗、传授犯罪方法、制作或者销售违禁物品、管制物品等违法犯罪活动的网站、通讯群组的；（二）发布有关制作或者销售毒品、枪支、淫秽物品等违禁物品、管制物品或者其他违法犯罪信息的；（三）为实施诈骗等违法犯罪活动发布信息的。 单位犯前款罪的，对单位判处罚金，并对其直接负责的主管人员和其他直接责任人员，依照第一款的规定处罚。 有前两款行为，同时构成其他犯罪的，依照处罚较重的规定定罪处罚。'
    },
    {
        'keywords': ['信用卡诈骗', '恶意透支', '信用卡被盗', '伪造信用卡', '盗刷信用卡',
                     '信用卡诈骗罪', '透支卡'],
        'pk': '中华人民共和国刑法_第一百九十六条',
        'law_name': '中华人民共和国刑法',
        'article': '第一百九十六条',
        'text': '有下列情形之一，进行信用卡诈骗活动，数额较大的，处五年以下有期徒刑或者拘役，并处二万元以上二十万元以下罚金；数额巨大或者有其他严重情节的，处五年以上十年以下有期徒刑，并处五万元以上五十万元以下罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处五万元以上五十万元以下罚金或者没收财产：（一）使用伪造的信用卡的；（二）使用以虚假的身份证明骗领的信用卡的；（三）冒用他人信用卡的；（四）恶意透支的。 前款所称恶意透支，是指持卡人以非法占有为目的，超过规定限额或者规定期限透支，并且经发卡银行催收后仍不归还的行为。 盗窃信用卡并使用的，依照本法第二百六十四条的规定定罪处罚。'
    },
    {
        'keywords': ['妨害信用卡管理', '非法持有信用卡', '买卖信用卡', '冒领信用卡',
                     '信用卡信息', '磁条信息'],
        'pk': '中华人民共和国刑法_第一百七十七条之一',
        'law_name': '中华人民共和国刑法',
        'article': '第一百七十七条之一',
        'text': '有下列情形之一，妨害信用卡管理的，处三年以下有期徒刑或者拘役，并处或者单处一万元以上十万元以下罚金；数量巨大或者有其他严重情节的，处三年以上七年以下有期徒刑，并处二万元以上十万元以下罚金：（一）明知是伪造的信用卡而持有、运输的，或者明知是伪造的空白信用卡而持有、运输的；（二）非法持有他人信用卡的；（三）使用虚假的身份证明骗领信用卡的；（四）出售、购买、为他人提供伪造的信用卡或者以虚假的身份证明骗领的信用卡的。'
    },
    {
        'keywords': ['盗窃罪', '盗窃财务', '入室盗窃', '偷窃', '盗窃公私财务'],
        'pk': '中华人民共和国刑法_第二百六十四条',
        'law_name': '中华人民共和国刑法',
        'article': '第二百六十四条',
        'text': '盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金；数额特别巨大或者有其他特别严重情节的，处十年以上有期徒刑或者无期徒刑，并处罚金或者没收财产。'
    },
    {
        'keywords': ['出借银行账户', '账户借用', '单位资金个人账户', '资金转个人账户',
                     '商业银行账户'],
        'pk': '中华人民共和国商业银行法_第四十八条',
        'law_name': '中华人民共和国商业银行法',
        'article': '第四十八条',
        'text': '企业事业单位可以自主选择一家商业银行的营业场所开立一个办理日常转账结算和现金收付的基本账户，不得开立两个以上基本账户。 任何单位和个人不得将单位的资金以个人名义开立账户存储。'
    },
    {
        'keywords': ['洗钱罪', '掩饰隐瞒犯罪所得', '毒黑私贪犯罪所得', '资金清洗', '转移犯罪所得'],
        'pk': '中华人民共和国刑法_第一百九十一条',
        'law_name': '中华人民共和国刑法',
        'article': '第一百九十一条',
        'text': '为掩饰、隐瞒毒品犯罪、黑社会性质的组织犯罪、恐怖活动犯罪、走私犯罪、贪污贿赂犯罪、破坏金融管理秩序犯罪、金融诈骗犯罪的所得及其产生的收益的来源和性质，依法追究刑事责任。'
    },
    {
        'keywords': ['骗取贷款罪', '高利转贷', '套取信贷资金', '违法发放贷款', '银行贷款犯罪'],
        'pk': '中华人民共和国刑法_第一百七十五条',
        'law_name': '中华人民共和国刑法',
        'article': '第一百七十五条',
        'text': '以转贷牟利为目的，套取金融机构信贷资金高利转贷他人，违法所得数额较大的，处三年以下有期徒刑或者拘役，并处违法所得一倍以上五倍以下罚金；数额巨大的，处三年以上七年以下有期徒刑，并处违法所得一倍以上五倍以下罚金。'
    },
]

STATUTE_KEYWORD_TRIE = {}
for entry in STATUTE_KEYWORDS:
    for kw in entry['keywords']:
        if kw not in STATUTE_KEYWORD_TRIE:
            STATUTE_KEYWORD_TRIE[kw] = entry['pk']

SEMANTIC_PATTERNS = [
    (re.compile(r'借[^\w\s]*(?:别人|他人|朋友|客户)?的?(?:银行)?卡'), '中华人民共和国刑法_第二百八十七条之一'),
    (re.compile(r'把(?:自己)?的?(?:银行)?卡借'), '中华人民共和国刑法_第二百八十七条之一'),
    (re.compile(r'出借(?:银行)?账户'), '中华人民共和国刑法_第二百八十七条之一'),
    (re.compile(r'借(?:银行)?账户'), '中华人民共和国刑法_第二百八十七条之一'),
    (re.compile(r'租借(?:银行)?(?:卡|账户)'), '中华人民共和国刑法_第二百八十七条之一'),
    (re.compile(r'(?:银行)?卡借给'), '中华人民共和国刑法_第二百八十七条之一'),
    (re.compile(r'可以把(?:银行)?卡'), '中华人民共和国刑法_第二百八十七条之一'),
]

def match_statute_keywords(query: str) -> List[Document]:
    """智能语义匹配：正则模式 + 关键词子串匹配，命中即强制召回"""
    matched_pks = set()
    docs = []
    for pattern, pk in SEMANTIC_PATTERNS:
        if pattern.search(query):
            matched_pks.add(pk)
    for kw, pk in STATUTE_KEYWORD_TRIE.items():
        if kw in query:
            matched_pks.add(pk)
    for entry in STATUTE_KEYWORDS:
        if entry['pk'] in matched_pks:
            docs.append(Document(
                page_content=entry['text'],
                metadata={
                    'source_type': 'statute',
                    'pk': entry['pk'],
                    'law_name': entry['law_name'],
                    'article': entry['article'],
                    'score': 0.0,
                }
            ))
    return docs


# ============================================================
# 1. 模型配置
# ============================================================
import os
FINETUNED_MODEL = r"E:\hermes_workspace\LAW_RAG\embedding_finetune\outputs\bge-base-zh-v1.5_20260529_1329"
CASES_MODEL = r"E:\hermes_workspace\LAW_RAG\models\BAAI\bge-large-zh-v1___5"
MILVUS_URI = "http://127.0.0.1:19530/LAW_RAG"

def get_embeddings_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=FINETUNED_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_cases_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CASES_MODEL)

def get_milvus_client():
    from pymilvus import MilvusClient
    return MilvusClient(uri=MILVUS_URI)

# ============================================================
# 2. 检索函数
# ============================================================
def retrieve_statutes(query: str, mc, embeddings_model, topk: int = 10) -> List[Document]:
    q_emb = embeddings_model.embed_query(query)
    res = mc.search(
        collection_name='statutes',
        data=[q_emb],
        limit=topk,
        search_params={'anns_field': 'embedding', 'params': {'ef': 256}},
        output_fields=['pk', 'law_name', 'article', 'text']
    )
    docs = []
    for hit in res[0]:
        entity = hit['entity']
        docs.append(Document(
            page_content=entity.get('text', ''),
            metadata={
                'source_type': 'statute',
                'pk': entity.get('pk', ''),
                'law_name': entity.get('law_name', ''),
                'article': entity.get('article', ''),
                'score': hit.get('score', 0.0),
            }
        ))
    return docs

def retrieve_cases(query: str, mc, cases_model, topk: int = 10) -> List[Document]:
    q_emb = cases_model.encode(query)
    res = mc.search(
        collection_name='cail_cases',
        data=[q_emb],
        limit=topk,
        search_params={'anns_field': 'vector', 'params': {'ef': 256}},
        output_fields=['case_id', 'source', 'fact', 'accusation', 'law_articles', 'term_months']
    )
    docs = []
    for hit in res[0]:
        entity = hit['entity']
        docs.append(Document(
            page_content=entity.get('fact', '')[:1500],
            metadata={
                'source_type': 'case',
                'case_id': entity.get('case_id', ''),
                'source': entity.get('source', ''),
                'fact': entity.get('fact', ''),
                'accusation': entity.get('accusation', ''),
                'law_articles': entity.get('law_articles', ''),
                'term_months': entity.get('term_months', 0),
                'score': hit.get('score', 0.0),
            }
        ))
    return docs

# ============================================================
# 3. RRF 融合
# ============================================================
def rrf_fusion(results_a: List[Document], results_b: List[Document], k: int = 60) -> List[Document]:
    scores = {}
    for rank, doc in enumerate(results_a):
        key = ('statute', doc.metadata.get('pk', ''))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc.metadata['rrf_score'] = scores[key]
        doc.metadata['rank_statutes'] = rank + 1
    for rank, doc in enumerate(results_b):
        key = ('case', doc.metadata.get('case_id', ''))
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        doc.metadata['rrf_score'] = scores[key]
        doc.metadata['rank_cases'] = rank + 1
    all_docs = results_a + results_b
    all_docs.sort(key=lambda d: d.metadata.get('rrf_score', 0.0), reverse=True)
    return all_docs

def hybrid_retrieve(query: str, mc, embeddings_model, cases_model, topk: int = 10):
    """
    混合检索：关键词召回 → 并行 statutes/cases 向量检索 → RRF融合 → TOP-K
    返回 (docs, timing_dict)
    """
    import time, concurrent.futures
    t_start = time.time()
    kw_docs = match_statute_keywords(query)
    t_kw = time.time()

    def _statutes():
        return retrieve_statutes(query, mc, embeddings_model, topk)

    def _cases():
        return retrieve_cases(query, mc, cases_model, topk)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_statutes = executor.submit(_statutes)
        f_cases = executor.submit(_cases)
        docs_a = f_statutes.result()
        docs_b = f_cases.result()

    t_search = time.time()
    kw_pks = {d.metadata['pk'] for d in kw_docs}
    docs_a_dedup = [d for d in docs_a if d.metadata['pk'] not in kw_pks]
    for d in kw_docs:
        d.metadata['rank_statutes'] = 0
    docs_a = kw_docs + docs_a_dedup

    fused = rrf_fusion(docs_a, docs_b)
    t_rrf = time.time()

    timing = {
        'keyword': t_kw - t_start,
        'statutes': t_search - t_kw,
        'cases': 0,
        'rrf': t_rrf - t_search,
        'total': t_rrf - t_start,
    }
    return fused[:topk], timing


# ============================================================
# 4. Context 构建
# ============================================================
def build_statutes_context(docs: List[Document]) -> str:
    statute_docs = [d for d in docs if d.metadata.get('source_type') == 'statute']
    if not statute_docs:
        return '（无法条参考）'
    result = []
    for i, doc in enumerate(statute_docs, 1):
        law_name = doc.metadata.get('law_name', '未知法律')
        article = doc.metadata.get('article', '未知条款')
        text = doc.page_content
        result.append(f"{i}. 【{law_name} 第{article}条】\n{text}")
    return '\n\n'.join(result)


def build_cases_context(docs: List[Document]) -> str:
    case_docs = [d for d in docs if d.metadata.get('source_type') == 'case']
    if not case_docs:
        return '（无相关案例）'
    result = []
    for i, doc in enumerate(case_docs, 1):
        acc = doc.metadata.get('accusation', '未知罪名')
        law_art = doc.metadata.get('law_articles', '未知法条')
        term = doc.metadata.get('term_months', 0)
        source = doc.metadata.get('source', '')
        fact = doc.page_content
        result.append(f"{i}. 【{acc} | {law_art} | {term}月】\n来源：{source}\n案情：{fact}")
    return '\n\n'.join(result)


def build_history_context(messages: list) -> str:
    """从会话历史构建上下文（最近10条）"""
    recent = messages[-10:] if len(messages) > 10 else messages
    if not recent:
        return '（无历史对话）'
    parts = []
    for msg in recent:
        role = '用户' if msg['role'] == 'user' else '助手'
        parts.append(f"{role}：{msg['content']}")
    return '\n'.join(parts)

# ============================================================
# 5. LLM 配置 & Prompt
# ============================================================
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '').strip()


def get_llm():
    if not DASHSCOPE_API_KEY:
        return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model='qwen3.6-plus',
        api_key=DASHSCOPE_API_KEY,
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        temperature=0,
    )

PROMPT_TEMPLATE = """你是一位专业的中国法律顾问。请严格基于检索到的法律信息回答，不要添加知识库以外的内容。


【相关法条】
{statutes_context}

【相关案例】
{cases_context}


【历史对话】
{history_context}


用户问题：{question}

回答要求：
1. **仅依据上述检索内容**：引用检索到的法条和案例，超出检索范围的不要臆测
2. 如果检索内容不足以回答，明确说明「根据当前检索信息，无法回答该问题」
3. 不要使用「根据我的知识」「一般来说」「通常来说」等基于自身知识的表达
4. 结合相关案例说明实际适用情况
5. 给出实际建议（如果适用）

回答："""
