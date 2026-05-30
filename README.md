# LAW_RAG 法律知识库问答系统

基于混合检索（关键词召回 + 双集合向量 + RRF 融合）的法律领域 RAG 系统，支持多轮对话和会话管理。

## 架构

```
LAW_RAG/
├── core/           # 核心检索引擎
│   └── rag.py      # 混合检索逻辑
├── db/             # 数据库模块
│   └── session_db.py  # PostgreSQL 会话存储
├── data/           # 数据处理
│   └── ingest.py   # 法律条文入库脚本
├── ui/             # Web UI
│   └── app.py      # Streamlit 前端
└── datasets/       # 法律原始数据（不上传）
    └── statutes/   # 58部法律 .docx 文件
```

## 检索流程

```
用户问题
   │
   ▼
┌─────────────────────────┐
│ 1. 关键词召回            │
│   - 正则模式匹配         │
│   - 子串匹配            │
│   命中则强制召回对应法条  │
└────────────┬────────────┘
             │ 并行
             ▼
┌─────────────────────────┐
│ 2. statutes 向量检索      │ ← 微调 bge-base-zh-v1.5 (dim=768)
│ 3. cail_cases 向量检索   │ ← bge-large-zh-v1.5 (dim=1024)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 4. RRF 融合 (k=60)      │
│   score = Σ 1/(k+rank)   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 5. TOP-K (K=10) 输出     │
└─────────────────────────┘
```

## 检索与重排方法

### 第一步：关键词召回

针对向量相似度不足但语义明确的query，绕过向量检索直接强制召回对应法条：

| 场景 | 召回法条 |
|------|----------|
| "借银行卡" / "帮信罪" | 刑法第287条之一 |
| "信用卡诈骗" / "恶意透支" | 刑法第196条 |
| "盗窃罪" | 刑法第264条 |
| "洗钱罪" | 刑法第191条 |

实现：正则模式匹配（如 `借[^\\w\\s]*的?(?:银行)?卡`）+ 关键词哈希表（`帮信罪` → `pk`）

### 第二步：双集合向量检索

- **statutes 集合**（法条）：微调 bge-base-zh-v1.5，dim=768，HNSW索引，COSINE度量
- **cail_cases 集合**（案例）：bge-large-zh-v1.5，dim=1024，HNSW索引，COSINE度量

两路检索通过 `ThreadPoolExecutor` 并行执行，节省 ~1s 延迟。

### 第三步：RRF 融合

Reciprocal Rank Fusion，k=60：

```
RRF_score(doc) = Σ 1/(k + rank_in_collection)
```

将 statutes 和 cail_cases 两路检索结果按 RRF_score 排序，得到统一候选列表。

### 第四步：TOP-K 输出

取融合后前 10 条作为最终检索结果，送入 LLM。

## 当前问题

1. **LLM 延迟 ~60s**：云端模型（qwen3.6-plus）推理速度是绝对瓶颈，无法本地优化
2. **关键词召回依赖人工维护**：无法穷举所有法律概念，需持续扩充关键词表
3. **法条 topk 无 rerank**：当前 statutes 召回 10 条后直接送入 LLM，未做语义重排
4. **语义检索召回率依赖向量模型质量**：fine-tuned 模型对特定概念（如帮信罪）效果有限

## 依赖

- Python 3.10+
- Milvus 2.4+ (http://127.0.0.1:19530)
- PostgreSQL 15+ (会话存储)
- 向量模型：
  - `embedding_finetune/outputs/bge-base-zh-v1.5_20260529_1329`（法条检索，微调）
  - `models/BAAI/bge-large-zh-v1.5`（案例检索）
- LLM：`qwen3.6-plus`（阿里百炼，需设置 `DASHSCOPE_API_KEY`）

## 启动

```bash
# 1. 启动 PostgreSQL
pg_ctl -D data/postgres start

# 2. 启动 Milvus
cd milvus && ./milvus run embedded

# 3. 入库法律条文（如需重新入库）
python data/ingest.py

# 4. 启动 Web UI
streamlit run ui/app.py
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里百炼 API Key（不设置则只展示检索结果） |

## 数据库

PostgreSQL 两张表：

```sql
sessions (session_id PK, title, created_at)
messages (id PK, session_id FK→sessions, role, content, created_at)
```

CASCADE 删除：删除会话时自动删除其所有消息。