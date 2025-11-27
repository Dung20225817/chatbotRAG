import streamlit as st
import os
import tempfile
import warnings
import logging

# Tắt warnings không cần thiết
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import shutil

# Cấu hình trang
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
    }
</style>
""", unsafe_allow_html=True)


class RAGChatbotApp:
    def __init__(self):
        self.vectorstore = None
        self.rag_chain = None
        self.embeddings = None
        self.llm = None
        
    def load_pdfs(self, uploaded_files):
        """Load và xử lý nhiều file PDF"""
        all_texts = []
        
        with st.spinner("🔄 Đang xử lý các file PDF..."):
            for uploaded_file in uploaded_files:
                # Lưu file tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Load PDF
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    # Thêm metadata về tên file
                    for doc in documents:
                        doc.metadata['source_file'] = uploaded_file.name
                    
                    # Chia nhỏ văn bản
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50,
                        length_function=len
                    )
                    texts = text_splitter.split_documents(documents)
                    all_texts.extend(texts)
                    
                    st.success(f"✅ {uploaded_file.name}: {len(texts)} chunks")
                    
                finally:
                    # Xóa file tạm
                    os.unlink(tmp_path)
        
        return all_texts
    
    def create_vectorstore(self, texts):
        """Tạo hoặc cập nhật vector store"""
        with st.spinner("🧠 Đang tạo embeddings..."):
            if self.embeddings is None:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="keepitreal/vietnamese-sbert",
                    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
                )
            
            # Xóa vector store cũ nếu có
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            
            # Tạo vector store mới
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            self.vectorstore.persist()
            
        st.success("✅ Vector store đã được tạo!")
    
    @st.cache_resource
    def load_llm(_self, model_name):
        """Load mô hình LLM (cache để không load lại)"""
        with st.spinner(f"🤖 Đang tải model {model_name}..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            
        st.success("✅ Model đã được tải!")
        return llm
    
    def format_docs(self, docs):
        """Format documents thành string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def setup_rag_chain(self, llm):
        """Thiết lập RAG chain với LCEL (LangChain Expression Language)"""
        
        # Template cho prompt
        template = """Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu không biết câu trả lời, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu".

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời chi tiết:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Tạo retriever
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Tạo RAG chain với LCEL
        self.rag_chain = (
            {
                "context": retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Lưu retriever để lấy source documents
        self.retriever = retriever
    
    def ask(self, question):
        """Đặt câu hỏi"""
        if self.rag_chain is None:
            return None
        
        # Lấy câu trả lời
        answer = self.rag_chain.invoke(question)
        
        # Lấy source documents
        source_docs = self.retriever.invoke(question)
        
        return {
            "result": answer,
            "source_documents": source_docs
        }


def main():
    # Header
    st.markdown('<div class="main-header">🤖 RAG Chatbot - Hỏi đáp từ PDF</div>', unsafe_allow_html=True)
    
    # Khởi tạo session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbotApp()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        # Chọn model
        model_options = {
            "VinALlama 7B (Tiếng Việt)": "vilm/vinallama-7b-chat",
            "Vistral 7B (Tiếng Việt)": "Viet-Mistral/Vistral-7B-Chat",
            "Mistral 7B (Đa ngôn ngữ)": "mistralai/Mistral-7B-Instruct-v0.2",
            "TinyLlama 1B (Nhẹ)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        }
        
        selected_model_name = st.selectbox(
            "Chọn mô hình LLM",
            options=list(model_options.keys()),
            help="Chọn model phù hợp với cấu hình máy của bạn"
        )
        selected_model = model_options[selected_model_name]
        
        st.divider()
        
        # Upload PDFs
        st.header("📁 Tải lên tài liệu")
        uploaded_files = st.file_uploader(
            "Chọn file PDF",
            type=['pdf'],
            accept_multiple_files=True,
            help="Bạn có thể tải lên nhiều file PDF cùng lúc"
        )
        
        if uploaded_files:
            st.write(f"**Đã chọn {len(uploaded_files)} file:**")
            for file in uploaded_files:
                st.write(f"- {file.name}")
        
        # Nút xử lý
        if st.button("🚀 Xử lý tài liệu", type="primary", use_container_width=True):
            if not uploaded_files:
                st.error("⚠️ Vui lòng tải lên ít nhất 1 file PDF")
            else:
                try:
                    # Load PDFs
                    texts = st.session_state.chatbot.load_pdfs(uploaded_files)
                    
                    # Tạo vector store
                    st.session_state.chatbot.create_vectorstore(texts)
                    
                    # Load LLM
                    if st.session_state.chatbot.llm is None:
                        st.session_state.chatbot.llm = st.session_state.chatbot.load_llm(selected_model)
                    
                    # Setup RAG chain
                    st.session_state.chatbot.setup_rag_chain(st.session_state.chatbot.llm)
                    
                    # Lưu danh sách file đã xử lý
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    
                    st.success("🎉 Sẵn sàng trả lời câu hỏi!")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
        
        st.divider()
        
        # Hiển thị thông tin
        st.header("ℹ️ Thông tin")
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"**Device:** {device}")
        
        if st.session_state.processed_files:
            st.success(f"**Đã xử lý:** {len(st.session_state.processed_files)} file")
            with st.expander("Xem danh sách file"):
                for fname in st.session_state.processed_files:
                    st.write(f"✓ {fname}")
        
        # Nút xóa lịch sử
        if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat area
    if not st.session_state.processed_files:
        st.markdown("""
        <div class="info-box">
        <h3>👋 Chào mừng bạn đến với RAG Chatbot!</h3>
        <p>Để bắt đầu:</p>
        <ol>
            <li>Tải lên file PDF ở sidebar bên trái</li>
            <li>Chọn mô hình LLM phù hợp</li>
            <li>Nhấn "Xử lý tài liệu"</li>
            <li>Bắt đầu đặt câu hỏi!</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Hiển thị lịch sử chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📄 Xem nguồn trích dẫn"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**Nguồn {i}:** {source['file']} (Trang {source['page']})")
                            st.caption(f"_{source['content'][:200]}..._")
        
        # Input chat
        if prompt := st.chat_input("Đặt câu hỏi về tài liệu..."):
            # Thêm câu hỏi vào lịch sử
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Lấy câu trả lời
            with st.chat_message("assistant"):
                with st.spinner("🤔 Đang suy nghĩ..."):
                    try:
                        result = st.session_state.chatbot.ask(prompt)
                        
                        if result:
                            answer = result["result"]
                            st.markdown(answer)
                            
                            # Lưu nguồn
                            sources = []
                            if result["source_documents"]:
                                for doc in result["source_documents"]:
                                    sources.append({
                                        "file": doc.metadata.get('source_file', 'Unknown'),
                                        "page": doc.metadata.get('page', 'N/A'),
                                        "content": doc.page_content
                                    })
                                
                                with st.expander("📄 Xem nguồn trích dẫn"):
                                    for i, source in enumerate(sources, 1):
                                        st.write(f"**Nguồn {i}:** {source['file']} (Trang {source['page']})")
                                        st.caption(f"_{source['content'][:200]}..._")
                            
                            # Thêm câu trả lời vào lịch sử
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })
                        else:
                            st.error("Không thể tạo câu trả lời. Vui lòng thử lại.")
                            
                    except Exception as e:
                        st.error(f"❌ Lỗi: {str(e)}")


if __name__ == "__main__":
    main()