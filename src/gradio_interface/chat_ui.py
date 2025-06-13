import gradio as gr
import tempfile
import os
import json
from pathlib import Path
from markdown import markdown
from langchain.schema import Document
from src.config import Config
from src.data_processing.loader import DataLoader
from src.data_processing.splitter import create_text_splitter
from src.rag_chain.retriever import create_prompt
from src.rag_chain.generator import create_llm
from src.data_processing.image_matcher import ImageMatcher
from langchain_community.vectorstores import Chroma  
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from transformers import AutoTokenizer, T5ForConditionalGeneration,T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
class ChatUI:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.llm = create_llm()
        self.prompt = create_prompt()        
        self.data_loader = DataLoader()
        self.image_matcher = ImageMatcher()
        self.history_file = "chat_history.json"
        self._load_history()
        self.text_splitter = create_text_splitter()
        self.css = """
        img { max-width: 300px; margin: 10px; border-radius: 10px; }
        .references { color: #666; font-size: 0.9em; }
        .warning { color: #ff4b4b; }
        """
        self.hybrid_retriever = self._init_hybrid_retriever()
    
    def _init_hybrid_retriever(self):
        """Khởi tạo hybrid retriever kết hợp vector và BM25"""
        try:
            dense_retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.7}
            )
            documents = self._get_all_documents()
            if not documents:
                print("Warning: No documents found for BM25 retriever")
                return dense_retriever
            # print("\n--- Dense Retriever Results ---")
            # test_query = "yykyukyuk"
            # dense_results = dense_retriever.invoke(test_query)
            
            # for i, doc in enumerate(dense_results):
            #     print(f"\nDocument {i+1}:")
            #     print(f"Content: {doc.page_content[:500]}...")  # In 200 ký tự đầu
            #     print(f"Score: {doc.metadata.get('score', 'N/A')}")    
            try:
                # Try using EnsembleRetriever
                sparse_retriever = BM25Retriever.from_documents(documents)
                # print("\n--- Sparse (BM25) Retriever Results ---")
                # sparse_results = sparse_retriever.invoke(test_query)
                # for i, doc in enumerate(sparse_results):
                    # print(f"\nDocument {i+1}:")
                    # print(f"Content: {doc.page_content[:500]}...")
                    # # BM25 score được tính trong quá trình retrieve
                    # print(f"BM25 Rank: {i+1}")
                # ensemble_retriever = EnsembleRetriever(
                #     retrievers=[dense_retriever, sparse_retriever],
                #     weights=[0.8, 0.2])
                # docs = ensemble_retriever.invoke(test_query)
                # print(f'7777777777777777777 \\n{docs}')
                return EnsembleRetriever(
                    retrievers=[dense_retriever, sparse_retriever],
                    weights=[0.8, 0.2]
                )
                
            except ImportError:
                # Fallback to basic weighted combination
                print("Falling back to basic weighted combination of retrievers")
                def combined_retriever(query):
                    dense_docs = dense_retriever.invoke(query)
                    sparse_docs = sparse_retriever.invoke(query)
                    
                    # Combine and deduplicate results
                    seen = set()
                    combined_docs = []
                    for doc in (dense_docs + sparse_docs):
                        if doc.page_content not in seen:
                            seen.add(doc.page_content)
                            combined_docs.append(doc)
                    
                    return combined_docs[:Config.RETRIEVE_TOP_K]
                    
                return combined_retriever
            
        except Exception as e:
            print(f"Error initializing hybrid retriever: {e}")
            return self.vector_store.as_retriever()  # Fallback to dense retriever

    def _generate_hyde_document(self, query):
        """Tạo pseudo-document với HyDE"""
        prompt = f"""
        Hãy viết một đoạn văn bản trả lời mẫu cho câu hỏi sau. 
        Đoạn văn nên chứa thông tin chính xác và tóm tắt câu trả lời cho câu hỏi không quá 500 ký tự.
        Đoạn văn này sẽ được sử dụng để tìm kiếm thông tin liên quan trong cơ sở dữ liệu của chúng tôi.:
        
        Câu hỏi: {query}
        """
        return self.llm.invoke(prompt)

    def _basic_ranking(self, query, documents):
        """Basic ranking using TF-IDF similarity"""
        
        # Create TF-IDF vectors
        tfidf = TfidfVectorizer()
        doc_vectors = tfidf.fit_transform([doc.page_content for doc in documents])
        query_vector = tfidf.transform([query])
        
        # Calculate similarities
        similarities = (query_vector @ doc_vectors.T).toarray()[0]
        
        # Sort documents by similarity
        sorted_pairs = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_pairs]

   
    def _load_history(self):
        try:
            with open(self.history_file, "r", encoding='utf-8') as f:
                self.chat_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.chat_history = []

    def _save_history(self):
        with open(self.history_file, "w",encoding='utf-8') as f:
            json.dump(self.chat_history, f, ensure_ascii=False, indent=2)
            
    def _process_uploaded_file(self, file):
        """Xử lý file upload và trả về documents"""
        try:
            # Tạo thư mục upload nếu chưa tồn tại
            os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
            
            # Xử lý file tạm từ Gradio
            if hasattr(file, "name"):
                temp_path = file.name
            else:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(file)
                    temp_path = tmp.name

            # Copy vào thư mục upload
            file_path = Path(Config.UPLOAD_DIR) / Path(file.name).name
            with open(temp_path, "rb") as src, open(file_path, "wb") as dst:
                dst.write(src.read())
                
            # Kiểm tra định dạng file
            if not str(file_path).lower().endswith(tuple(Config.ALLOWED_FILE_TYPES)):
                os.remove(file_path)
                raise ValueError(f"Loại file không được hỗ trợ. Chỉ chấp nhận: {', '.join(Config.ALLOWED_FILE_TYPES)}")
            
            # Load documents
            documents = self.data_loader.load_file(file_path)
            
            # Xóa file tạm và file upload sau khi xử lý xong
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return documents

        except Exception as e:
            # Cleanup nếu có lỗi
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            raise e

    def _process_web_url(self, url):
        """Xử lý URL website"""
        if not any(domain in url for domain in Config.ALLOWED_DOMAINS):
            raise ValueError(f"Chỉ hỗ trợ các website: {', '.join(Config.ALLOWED_DOMAINS)}")
        return self.data_loader.load_web(url)

    def handle_update(self, file, url):
        """Xử lý cập nhật dữ liệu từ file và URL"""
        try:
            documents = []
            
            # Xử lý file upload
            if file is not None:
                documents += self._process_uploaded_file(file)
            
            # Xử lý URL
            if url.strip():
                documents += self._process_web_url(url.strip())
            
            if not documents:
                return "⚠️ Vui lòng chọn file hoặc nhập URL hợp lệ"
            
            split_docs = self.text_splitter.split_documents(documents)
            self.vector_store.add_documents(split_docs)
            
            return "✅ Cập nhật dữ liệu thành công!"
        except Exception as e:
            print(f"Error in handle_update: {str(e)}")
            return f"⚠️ Lỗi: {str(e)}"

    def format_response(self, response, docs, query):
        """Định dạng câu trả lời với hình ảnh và nguồn dẫn"""
        # Format markdown cho response
        md_content = f"{response}\n\n"
        
        # Tìm ảnh phù hợp dựa trên cả câu hỏi và câu trả lời
        combined_query = f"{query} {response}"
        matched_images = self.image_matcher.find_matching_images(
            query=combined_query,
            threshold=1  # Yêu cầu ít nhất 1 cụm từ khớp
        )
        print(f"Found {len(matched_images)} matching images")
        
        if matched_images:
            md_content += "## 📸 Hình ảnh liên quan\n"
            for img in matched_images[:Config.MAX_IMAGES_TO_SHOW]:
                print(f"Adding image: {img['name']} (matched words: {img['matched_words']})")
                # Chuyển đổi đường dẫn thành file:// URI
                img_path = Path(img['path']).absolute().as_uri()
                md_content += f"![{img['name']}]({img_path})\n"
        
        return markdown(md_content)

    def chat(self, message, chat_history):
        """Xử lý chat message"""
        try:
            # Convert chat history từ gradio format
            history_message = []
            for user_msg, bot_msg in chat_history:
                history_message.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": bot_msg}
                ])
                

            # Bước 1: Tạo HyDE document
            hyde_document = self._generate_hyde_document(message)
            print("11111111111111111111111111111111111111111111")
            # print(f"HyDE llm trả lời ko dùng retrived:\\n {hyde_document}...")
            #Bước 2: Hybrid retrieval sử dụng HyDE document
            docs = self.hybrid_retriever.invoke(hyde_document)
            print("222222222222222222222222222222222222222222222")
            # print(f"Retrieved lấy k vector data tương đồng với câu trả lời HyDE \\n{docs} ")
            # Bước 3: Rerank documents với monoT5
            if docs:
                docs = self._basic_ranking(message, docs)[:Config.RETRIEVE_TOP_K]
            # Tạo context
            context = "\n\n".join([d.page_content for d in docs]) if docs else ""
            print("333333333333333333333333333333333333333333333")
            # print(F'=========Tạo Context:\\n{context}==========')
            # Tạo prompt
            formatted_prompt = self.prompt.format(
                context=context,
                question=message,
                chat_history=history_message
            )
            
            # Tạo response
            response = self.llm.invoke(formatted_prompt)
            # Format response
            formatted_response = self.format_response(response, docs, query=message)
            
            # Cập nhật lịch sử chat
            chat_history.append((message, formatted_response))
            
            # Save to file 
            self.chat_history = history_message + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": formatted_response}
            ]
            self._save_history()
            
            return "", chat_history
        except Exception as e:
            error_msg = f"⚠️ Lỗi hệ thống: {str(e)}"
            chat_history.append((message, error_msg))
            return "", chat_history

        

    def create_interface(self):
        """Tạo giao diện Gradio"""
        with gr.Blocks(css=self.css, theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🗺️ Trợ lý Du lịch Thông minh")
            
            with gr.Row():
                with gr.Column(scale=3):
                    # Convert saved history to list pairs format
                    initial_history = []
                    for i in range(0, len(self.chat_history)-1, 2):
                        user_msg = self.chat_history[i]["content"]
                        bot_msg = self.chat_history[i+1]["content"]
                        initial_history.append([user_msg, bot_msg])
                    
                    chatbot = gr.Chatbot(
                        value=initial_history,
                        label="Chat History",
                        height=600,
                        avatar_images=(
                            "https://cdn-icons-png.flaticon.com/512/1995/1995515.png",
                            "https://cdn-icons-png.flaticon.com/512/4712/4712035.png" 
                        )
                    )
                    msg = gr.Textbox(
                        label="Nhập câu hỏi về du lịch",
                        placeholder="Ví dụ: Kể tên các khách sạn tốt ở Đà Nẵng..."
                    )
                    clear_btn = gr.Button("Xóa lịch sử", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("## 🗂️ Quản lý dữ liệu")
                    with gr.Tab("Upload File"):
                        file_upload = gr.File(
                            label="Tải lên tài liệu",
                            file_types=Config.ALLOWED_FILE_TYPES
                        )
                    with gr.Tab("Website"):
                        url_input = gr.Textbox(
                            label="Nhập URL website du lịch",
                            placeholder="Ví dụ: https://dulichvietnam.com/..."
                        )
                    update_btn = gr.Button("Cập nhật dữ liệu", variant="primary")
                    status = gr.Markdown()
            
            # Event handlers
            msg.submit(
                self.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            update_btn.click(
                self.handle_update,
                inputs=[file_upload, url_input],
                outputs=[status]
            )
            
            clear_btn.click(
                fn=self.clear_chat_history,
                inputs=None,
                outputs=[chatbot],
                api_name="clear_history"
            )
        return demo
    
    def clear_chat_history(self):
        """Xóa lịch sử chat"""
        # Xóa file lưu history
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
        
        # Reset chat history trong memory
        self.chat_history = []
        
        # Return empty list để clear UI
        return []

    def _get_all_documents(self):
        """Lấy tất cả documents từ vector store"""
        collection = self.vector_store.get()
        documents = []
        if collection and "documents" in collection and "metadatas" in collection:
            for doc, meta in zip(collection["documents"], collection["metadatas"]):
                documents.append(
                    Document(
                        page_content=doc,
                        metadata=meta
                    )
                )
        return documents

def create_chat_interface(vector_store):
    chat_ui = ChatUI(vector_store)
    return chat_ui.create_interface()