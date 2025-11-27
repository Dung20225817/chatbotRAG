🤖 RAG Chatbot PDF – Vietnamese Document Q&A System
<p align="center"> <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white" /> <img src="https://img.shields.io/badge/LLM-HuggingFace-blue?logo=huggingface" /> <img src="https://img.shields.io/badge/RAG-LangChain-1C3C3C?logo=chainlink&logoColor=white" /> <img src="https://img.shields.io/badge/VectorDB-ChromaDB-3DDC84" /> <img src="https://img.shields.io/badge/Embeddings-SBERT-orange" /> </p>

RAG Chatbot PDF là ứng dụng giúp bạn trò chuyện với file PDF bằng tiếng Việt, sử dụng công nghệ RAG (Retrieval-Augmented Generation) để tạo câu trả lời chính xác và có trích dẫn nguồn theo trang.

Ứng dụng hỗ trợ nhiều mô hình LLM, xử lý nhiều file PDF cùng lúc và cho phép truy vấn nội dung cực nhanh nhờ Vector Search (ChromaDB + SBERT).
## 🚀 Tính năng chính

### 📄 1. Upload PDF  
- Hỗ trợ nhiều file cùng lúc  
- Tự động tách trang và trích xuất văn bản  
- Dùng `PyPDF2` hoặc `pdfplumber` để đọc nội dung  

### 🧠 2. Xây dựng vector database  
- Dùng Sentence-BERT hoặc mô hình embedding khác  
- Lưu trữ toàn bộ văn bản dưới dạng vector  
- Tìm kiếm nhanh bằng cosine similarity  

### 🔍 3. Hỏi – Đáp (Q&A) bằng RAG  
- Người dùng đặt câu hỏi  
- Hệ thống tìm đoạn văn phù hợp nhất từ PDF  
- Kết hợp với LLM để sinh câu trả lời tự nhiên, chính xác  

### 🤖 4. Tích hợp LLM HuggingFace  
- Hỗ trợ bất kỳ mô hình LLM nào trên HuggingFace  
- Tự động load bằng `transformers` + pipeline  
- Chạy được GPU (CUDA) hoặc CPU  

### 📊 5. Giao diện Streamlit  
- Trực quan, dễ sử dụng  
- Spinner & progress để theo dõi tiến độ load model và xử lý PDF  

