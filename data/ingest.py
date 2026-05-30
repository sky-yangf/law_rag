"""
LAW_RAG 入库脚本：将 datasets/statutes 下所有 docx 法律文件按条切块入库
目标数据库：LAW_RAG
集合名：statutes
字段：pk(primary key), law_name(index), article(index), text(no-index), embedding(vector768)
索引：embedding-HNSW(COSINE), law_name-标量, article-标量
"""
import os, re
from docx import Document
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client.index import IndexParams


# ── 配置 ────────────────────────────────────────────────────────────────
DB_URI           = "http://127.0.0.1:19530"
DB_NAME          = "LAW_RAG"
COLLECTION_NAME  = "statutes"
STATUTES_DIR     = r"E:\hermes_workspace\LAW_RAG\datasets\statutes"
EMBEDDING_MODEL  = r"E:\hermes_workspace\LAW_RAG\embedding_finetune\outputs\bge-base-zh-v1.5_20260529_1329"
VECTOR_DIM = 768

# ── 加载模型 ────────────────────────────────────────────────────────────
print("Loading embedding model ...")
emb_model = SentenceTransformer(EMBEDDING_MODEL)
print(f"  dim={emb_model.get_sentence_embedding_dimension()}")


print("Connecting to Milvus ...")
mc = MilvusClient(uri=DB_URI)
mc.using_database(DB_NAME)

# ── 删除旧集合（如有）───────────────────────────────────────────────────
if mc.has_collection(COLLECTION_NAME):
    print(f"Dropping existing collection '{COLLECTION_NAME}' ...")
    mc.drop_collection(COLLECTION_NAME)


# ── 创建集合 ───────────────────────────────────────────────────────────
print(f"Creating collection '{COLLECTION_NAME}' ...")
from pymilvus.orm.schema import CollectionSchema, FieldSchema


fields = [
    FieldSchema(name="pk",        dtype=DataType.VARCHAR, max_length=256, is_primary=True),
    FieldSchema(name="law_name",  dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="article",   dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="text",      dtype=DataType.VARCHAR, max_length=8192),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
]
schema = CollectionSchema(fields=fields, description="常用法律条文（58部）")

index_params = IndexParams()
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 200},
)
index_params.add_index(field_name="law_name", index_type="STL_SORT")
index_params.add_index(field_name="article", index_type="STL_SORT")

mc.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema,
    index_params=index_params,
)
print("Collection created with indexes.")


# ── 读取所有 docx 文件 ──────────────────────────────────────────────────
def extract_law_name(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r'_\d{8}$', '', name)
    return name

def chunk_by_article(doc):
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    chunks = []
    current_article = None
    current_text = []

    for para in paragraphs:
        m = re.match(r'^(第[一二三四五六七八九十百千零〇]+条[^\s]*)\s*(.*)', para)
        if m:
            if current_article and current_text:
                chunks.append((current_article, ' '.join(current_text)))
            current_article = m.group(1)
            current_text = [m.group(2)] if m.group(2) else []
        elif current_article:
            current_text.append(para)

    if current_article and current_text:
        chunks.append((current_article, ' '.join(current_text)))

    return chunks

# ── 批量入库 ───────────────────────────────────────────────────────────
print("\nProcessing docx files ...")
docx_files = [f for f in os.listdir(STATUTES_DIR) if f.endswith('.docx')]
docx_files.sort()

all_rows = []
total_chunks = 0
law_counts = {}

for fname in docx_files:
    fpath = os.path.join(STATUTES_DIR, fname)
    law_name = extract_law_name(fname)
    try:
        doc = Document(fpath)
        chunks = chunk_by_article(doc)
    except Exception as e:
        print(f"  [{law_name}] ERROR: {e}")
        continue

    print(f"  [{law_name}] {len(chunks)} 条")
    law_counts[law_name] = len(chunks)

    for article, text in chunks:
        if len(text) < 10:
            continue
        pk = f"{law_name}_{article}"
        all_rows.append({"pk": pk, "law_name": law_name, "article": article, "text": text})
        total_chunks += 1

print(f"\nTotal chunks to ingest: {total_chunks} ({len(law_counts)} laws)")


# ── 生成向量 ───────────────────────────────────────────────────────────
print("Generating embeddings ...")
texts = [r["text"] for r in all_rows]
batch_size = 128
embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    emb_batch = emb_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    for emb in emb_batch:
        embeddings.append(emb.tolist())
    print(f"  {min(i+batch_size, len(texts))}/{len(texts)} done", flush=True)


for row, emb in zip(all_rows, embeddings):
    row["embedding"] = emb

# ── 写入 Milvus ────────────────────────────────────────────────────────
print(f"\nInserting into Milvus ...")
result = mc.insert(collection_name=COLLECTION_NAME, data=all_rows)
print(f"  Inserted {len(all_rows)} rows.")

# ── 加载集合 ────────────────────────────────────────────────────────────
print("Loading collection ...")
mc.load_collection(COLLECTION_NAME)

# ── 验证 ──────────────────────────────────────────────────────────────
stats = mc.get_collection_stats(COLLECTION_NAME)
print(f"\nCollection stats: {stats}")
idx_list = mc.list_indexes(COLLECTION_NAME)
print(f"Indexes: {idx_list}")

print("\n各法律入库条数：")
for law, cnt in sorted(law_counts.items(), key=lambda x: -x[1]):
    print(f"  {law}: {cnt}")

print("\n=== 入库完成 ===")
print(f"  集合：{COLLECTION_NAME}")
print(f"  数据库：{DB_NAME}")
print(f"  总条数：{stats['row_count']}")
