import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# ── 페이지 설정
st.set_page_config(
    page_title="웹툰 산업 리포트 Q&A 챗봇",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.chat-user {
    background: #e8f0fe; border-radius: 12px 12px 2px 12px;
    padding: 10px 16px; margin: 6px 0; text-align: right; color: #1a1a2e;
}
.chat-bot {
    background: #f1f3f4; border-radius: 12px 12px 12px 2px;
    padding: 10px 16px; margin: 6px 0; color: #1a1a2e;
}
.source-box {
    background: #fff8e1; border-left: 3px solid #f9a825;
    padding: 8px 12px; border-radius: 4px; font-size: 0.8rem; color: #555; margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None

@st.cache_resource(show_spinner="📚 PDF를 분석하는 중입니다...")
def build_vectorstore(file_bytes_list, file_names):
    all_docs = []
    for file_bytes, file_name in zip(file_bytes_list, file_names):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_name
        all_docs.extend(docs)
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def build_chain(vectorstore, api_key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.3,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template("""당신은 웹툰 산업 전문 AI 어시스턴트입니다.
아래 제공된 문서(KOCCA 웹툰 실태조사 보고서)를 기반으로 질문에 답변해주세요.
문서에 없는 내용은 '보고서에서 해당 내용을 찾을 수 없습니다'라고 말해주세요.
답변은 한국어로 친절하고 명확하게 해주세요.

문서 내용:
{context}

질문: {question}

답변:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── 메인 UI
st.markdown("# 🎨 웹툰 산업 리포트 Q&A 챗봇")
st.markdown("KOCCA 웹툰 실태조사 보고서를 업로드하고 질문해보세요")

api_key = get_api_key()
if not api_key:
    st.error("⚠️ GROQ_API_KEY가 설정되지 않았습니다.")
    st.stop()

with st.sidebar:
    st.header("📁 PDF 업로드")
    uploaded_files = st.file_uploader(
        "KOCCA 웹툰 보고서 PDF를 업로드하세요",
        type=["pdf"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)}개 파일 업로드 완료")
        for f in uploaded_files:
            st.caption(f"📄 {f.name}")

    st.divider()
    st.markdown("**예시 질문**")
    example_questions = [
        "웹툰 시장 규모는 얼마나 되나요?",
        "웹툰 작가의 평균 수입은?",
        "웹툰 플랫폼 현황을 알려주세요",
        "웹툰 작가가 겪는 어려움은 무엇인가요?",
        "웹툰 IP OSMU 현황은?",
    ]
    for q in example_questions:
        if st.button(q, key=q):
            st.session_state["input_question"] = q

    st.divider()
    if st.button("🗑️ 대화 초기화"):
        st.session_state["chat_history"] = []
        st.rerun()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if uploaded_files:
    file_bytes_list = [f.read() for f in uploaded_files]
    file_names = [f.name for f in uploaded_files]
    vectorstore = build_vectorstore(tuple(file_bytes_list), tuple(file_names))
    chain, retriever = build_chain(vectorstore, api_key)

    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                sources_text = " | ".join(set(msg["sources"]))
                st.markdown(f'<div class="source-box">📌 출처: {sources_text}</div>', unsafe_allow_html=True)

    default_q = st.session_state.pop("input_question", "")
    user_input = st.chat_input("웹툰 산업에 대해 질문해보세요...")

    if not user_input and default_q:
        user_input = default_q

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("🔍 보고서에서 답변을 찾는 중..."):
            answer = chain.invoke(user_input)
            source_docs = retriever.invoke(user_input)
        sources = list(set([
            f"{doc.metadata.get('source', '알 수 없음')} p.{doc.metadata.get('page', '?') + 1}"
            for doc in source_docs
        ]))
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.rerun()

else:
    st.info("👈 왼쪽 사이드바에서 KOCCA 웹툰 보고서 PDF를 업로드해주세요.")
    st.markdown("""
    ### 사용 방법
    1. 사이드바에서 PDF 파일을 업로드합니다
    2. 업로드가 완료되면 질문창이 활성화됩니다
    3. 웹툰 산업에 관한 질문을 입력하세요
    4. AI가 보고서를 기반으로 답변하고 출처 페이지를 알려드립니다
    """)