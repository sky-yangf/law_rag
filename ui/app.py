"""
LAW_RAG Streamlit Web UI
依赖：
  core.rag   - RAG 检索引擎
  db.session_db - PostgreSQL 会话管理
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from core.rag import (
    get_embeddings_model, get_cases_model, get_milvus_client,
    hybrid_retrieve, build_statutes_context, build_cases_context,
    build_history_context, get_llm, PROMPT_TEMPLATE
)
from db.session_db import (
    init_db, list_sessions, create_session, delete_session,
    load_messages, save_message, update_session_title
)

st.set_page_config(page_title="LAW_RAG 法律问答", page_icon="⚖️", layout="wide")
st.title("⚖️ LAW_RAG 法律问答系统")

init_db()

# ————————————————————————————————————————————————————————————————
# 侧边栏：会话管理
# ————————————————————————————————————————————————————————————————
with st.sidebar:
    st.markdown("### 💬 会话管理")
    sessions = list_sessions()

    if st.button("➕ 新会话", use_container_width=True):
        new_sid = create_session("新会话")
        st.session_state.session_id = new_sid
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    for s in sessions:
        sid = s['session_id']
        label = s['title'][:20] + ('...' if len(s['title']) > 20 else '')
        ts = s['created_at'].strftime('%m-%d %H:%M')
        col1, col2 = st.columns([1, 4])
        with col1:
            marker = "🟢" if st.session_state.get('session_id') == sid else "  "
            st.markdown(f"{marker} `{ts}`")
        with col2:
            if st.button(label, key=f"sid_{sid}", use_container_width=True):
                st.session_state.session_id = sid
                st.session_state.messages = load_messages(sid)
                st.rerun()

    st.markdown("---")

    if st.session_state.get('session_id'):
        if st.button("🗑️ 删除当前会话", use_container_width=True):
            cur_sid = st.session_state['session_id']
            delete_session(cur_sid)
            remaining = list_sessions()
            if remaining:
                st.session_state.session_id = remaining[0]['session_id']
                st.session_state.messages = load_messages(st.session_state.session_id)
            else:
                st.session_state.session_id = create_session("新会话")
                st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("检索：statutes + cail_cases (RRF)")
    st.caption("模型：qwen3.6-plus")

# ————————————————————————————————————————————————————————————————
# 会话状态初始化
# ————————————————————————————————————————————————————————————————
if 'session_id' not in st.session_state:
    remaining = list_sessions()
    if remaining:
        st.session_state.session_id = remaining[0]['session_id']
        st.session_state.messages = load_messages(remaining[0]['session_id'])
    else:
        st.session_state.session_id = create_session("新会话")
        st.session_state.messages = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

# ————————————————————————————————————————————————————————————————
# 渲染聊天历史
# ————————————————————————————————————————————————————————————————
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg['content'])
    else:
        with st.chat_message("assistant", avatar="⚖️"):
            st.markdown(msg['content'])

# ————————————————————————————————————————————————————————————————
# 用户输入
# ————————————————————————————————————————————————————————————————
if prompt := st.chat_input("请输入法律问题..."):
    current_sid = st.session_state.session_id
    save_message(current_sid, 'user', prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    if len(st.session_state.messages) == 1:
        title = prompt[:30] + ('...' if len(prompt) > 30 else '')
        update_session_title(current_sid, title)

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    import time
    t0 = time.time()

    try:
        mc = get_milvus_client()
        emb = get_embeddings_model()
        cases_m = get_cases_model()
        docs, retrieve_timing = hybrid_retrieve(prompt, mc, emb, cases_m, topk=10)
        statutes_ctx = build_statutes_context(docs)
        cases_ctx = build_cases_context(docs)
        history_ctx = build_history_context(st.session_state.messages[:-1])

        llm = get_llm()
        t_llm_start = time.time()
        if llm:
            final_prompt = PROMPT_TEMPLATE.format(
                statutes_context=statutes_ctx,
                cases_context=cases_ctx,
                history_context=history_ctx,
                question=prompt
            )
            with st.chat_message("assistant", avatar="⚖️"):
                placeholder = st.empty()
                full_answer = ""
                for chunk in llm.stream(final_prompt):
                    token = chunk.content if hasattr(chunk, 'content') else ""
                    full_answer += token
                    placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)
                answer = full_answer
        else:
            answer = f"⚠️ 未配置 DASHSCOPE_API_KEY，以下是检索到的相关法条：\n\n{statutes_ctx}\n\n{cases_ctx}"
        llm_time = time.time() - t_llm_start

    except Exception as e:
        answer = f"❌ 检索出错：{str(e)}"
        retrieve_timing = {'keyword': 0, 'statutes': 0, 'cases': 0, 'rrf': 0, 'total': 0}
        llm_time = 0

    total_time = time.time() - t0

    save_message(current_sid, 'assistant', answer)
    st.session_state.messages.append({'role': 'assistant', 'content': answer})


    timing_line = (
        f"⏱️ 关键词 {retrieve_timing['keyword']:.1f}s · "
        f"法条检索 {retrieve_timing['statutes']:.1f}s · "
        f"案例检索 {retrieve_timing['cases']:.1f}s · "
        f"RRF融合 {retrieve_timing['rrf']:.1f}s · "
        f"LLM {llm_time:.1f}s · "
        f"总耗时 {total_time:.1f}s"
    )

    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown(answer)
        st.caption(timing_line)

    with st.expander("📂 检索来源"):
        statute_docs = [d for d in docs if d.metadata.get('source_type') == 'statute']
        case_docs = [d for d in docs if d.metadata.get('source_type') == 'case']


        st.markdown("**【法条来源】**")
        if statute_docs:
            for i, doc in enumerate(statute_docs[:5], 1):
                law = doc.metadata.get('law_name', '?')
                article = doc.metadata.get('article', '?')
                score = doc.metadata.get('score', 0.0)
                rank = doc.metadata.get('rank_statutes', '?')
                tag = ' ★关键词' if rank == 0 else ''
                text = doc.page_content[:120]
                st.markdown(f"`{i}. [{law} 第{article}条] score={score:.3f}{tag}`")
                st.markdown(f"&nbsp;&nbsp;{text}...")
        else:
            st.markdown("（无法条命中）")


        if case_docs:
            st.markdown("**【案例来源】**")
            for i, doc in enumerate(case_docs[:3], 1):
                acc = doc.metadata.get('accusation', '?')
                law_art = doc.metadata.get('law_articles', '?')
                term = doc.metadata.get('term_months', 0)
                score = doc.metadata.get('score', 0.0)
                st.markdown(f"`{i}. [{acc} | {law_art} | {term}月] score={score:.3f}`")
                st.markdown(f"&nbsp;&nbsp;{doc.page_content[:120]}...")
