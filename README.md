🤖 RAG Chatbot – Hỏi đáp thông minh từ tài liệu PDF

RAG Chatbot là một ứng dụng Streamlit cho phép bạn trò chuyện với tài liệu PDF bằng tiếng Việt.
Hệ thống kết hợp LLM (Large Language Model) và RAG (Retrieval-Augmented Generation) để trích xuất thông tin chính xác từ tài liệu mà bạn cung cấp.

Ứng dụng phù hợp cho:

Sinh viên muốn hỏi bài từ giáo trình

Doanh nghiệp muốn tra cứu tài liệu nội bộ

Nhà nghiên cứu phân tích báo cáo

Bất kỳ ai muốn truy vấn tài liệu nhanh hơn thay vì phải đọc toàn bộ

⭐ Tính năng chính
📁 1. Tải lên & xử lý nhiều file PDF

Hỗ trợ nhiều file cùng lúc

Tự động phân trang & chia văn bản thành các đoạn nhỏ (chunking)

Lưu metadata như tên file & số trang để hiển thị nguồn gốc câu trả lời

🔍 2. Xây dựng Vector Store với SBERT tiếng Việt

Dùng mô hình keepitreal/vietnamese-sbert để tạo embeddings chính xác

Lưu trữ bằng ChromaDB, cho phép truy xuất nhanh

Tự động xóa database cũ khi nạp tài liệu mới

🧠 3. Hỗ trợ nhiều mô hình LLM mạnh

Cho phép chọn các model như:

VinALlama 7B

Vistral 7B Chat

Mistral 7B Instruct

TinyLlama 1B

Model được load bằng HuggingFace Transformers và cache lại để không tải lại mỗi lần.

🔎 4. RAG Pipeline thông minh (LangChain LCEL)

Ứng dụng sử dụng:

Retriever để tìm đoạn văn liên quan

Custom PromptTemplate để điều khiển mô hình

LangChain Expression Language để kết nối các bước

StrOutputParser để trả kết quả sạch & dễ đọc

💬 5. Giao diện chat mượt mà bằng Streamlit

Lưu lịch sử hội thoại

Hiển thị nguồn trích dẫn theo từng câu trả lời

Tùy chỉnh giao diện bằng CSS

Tự động hiển thị file đã xử lý và thông tin thiết bị (CPU/GPU)

🚀 Luồng hoạt động của hệ thống

Người dùng tải lên 1 hoặc nhiều file PDF

Hệ thống trích xuất nội dung → chia đoạn

Tạo vector embeddings bằng SBERT

Lưu vào ChromaDB

Người dùng đặt câu hỏi

Retriever lấy 3 đoạn liên quan nhất (top-k)

LLM sinh câu trả lời dựa trên ngữ cảnh

Trả về câu trả lời + nguồn (page, file)
