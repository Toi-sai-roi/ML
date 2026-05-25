# HỆ THỐNG PHÂN LOẠI TIN NHẮN SPAM ĐA NGÔN NGỮ (ANH - VIỆT) - BẢN CẢI TIẾN CÔNG NGHỆ

Dự án ứng dụng Học máy (Machine Learning) và Xử lý ngôn ngữ tự nhiên (NLP) nhằm tự động nhận diện và phân loại một đoạn văn bản (tin nhắn) là **Spam** (tin nhắn rác, quảng cáo, lừa đảo) hay **Ham** (tin nhắn bình thường). 

Đây là phiên bản nâng cấp toàn diện về cả thuật toán xử lý dữ liệu lẫn trải nghiệm giao diện người dùng.

---

## 1. Các Cải Tiến Công Nghệ Đột Phá (So Với Bản Cũ)

* **Hỗ trợ đa ngôn ngữ (Hỗn hợp Anh - Việt):** Không còn bó hẹp ở tập dữ liệu tiếng Anh thô, mô hình đã được huấn luyện đồng thời trên cả tập dữ liệu tiếng Việt thực tế (tin nhắn lừa đảo trúng thưởng, tìm việc làm, vay vốn, tin nhắn sinh hoạt...).
* **Cân bằng dữ liệu (Data Balancing) bằng Down-sampling:** Khắc phục triệt để hiện tượng học vẹt (Overfitting) do tập dữ liệu UCI gốc bị lệch (86.6% Ham). Hệ thống tự động lấy mẫu ngẫu nhiên để đưa tỷ lệ Spam/Ham về mức cân bằng **1:1** hoàn hảo.
* **Tối ưu hóa NLP cho Tiếng Việt:** Loại bỏ hoàn toàn bộ cắt đuôi từ (*PorterStemmer*) cũ của NLTK. Cải tiến này giúp giữ nguyên cấu trúc ngữ nghĩa tự nhiên của từ vựng tiếng Việt không bị méo mó, tăng mạnh độ chính xác khi phân loại thực tế.
* **Đóng gói End-to-End Pipeline vật lý:** Hàm tiền xử lý chuỗi được tích hợp trực tiếp vào tham số `preprocessor` của bộ trích xuất `TfidfVectorizer` bên trong Scikit-learn Pipeline. Toàn bộ thực thể được nén lại thành duy nhất một file `spam_classifier.pkl`, triệt tiêu rủi ro lệch logic dữ liệu (Feature Mismatch) khi chạy app.
* **Nâng cấp tính năng Upload File:** Giao diện Web được trang bị thêm tính năng bóc tách dữ liệu từ file `.txt`, `.csv` để tiến hành phân loại hàng loạt (Bulk Classification).

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
├── .venv/                  # Môi trường ảo Python (Virtual Environment)
├── spam_vietnamese.csv     # Tập dữ liệu tin nhắn mẫu bằng Tiếng Việt
├── SMSSpamCollection       # Tập dữ liệu tin nhắn Tiếng Anh từ UCI (Tự động tải)
├── train.py                # Script huấn luyện, so sánh mô hình và đóng gói Pipeline
├── test.py                 # Script kiểm thử nhanh 10 test case Anh - Việt trên Terminal
├── app.py                  # Mã nguồn giao diện Web UI cao cấp (Streamlit)
├── spam_classifier.pkl     # File mô hình nhị phân đóng gói (Sinh ra sau khi train)
├── requirements.txt        # Danh sách các thư viện cấu hình bắt buộc
└── README.md               # Tài liệu hướng dẫn dự án (Chính là file này)

```

---

## 3. Kiến Trúc Luồng Xử Lý (Pipeline)

Hệ thống hoạt động dựa trên luồng xử lý khép kín tuần tự:

1. **Thu thập & Hợp nhất dữ liệu:** Tải tập dữ liệu UCI $\rightarrow$ Cân bằng mẫu 1:1 $\rightarrow$ Gộp dữ liệu tiếng Việt $\rightarrow$ Loại bỏ các dòng trùng lặp (`drop_duplicates`).
2. **Tiền xử lý văn bản thô:** Chuyển chữ thường (Lowercase) $\rightarrow$ Loại bỏ dấu câu/ký tự đặc biệt $\rightarrow$ Tách từ (Tokenize) $\rightarrow$ Loại bỏ từ dừng (Stopwords).
3. **Trích xuất đặc trưng toán học:** Dùng **TF-IDF Vectorizer** cấu hình mở rộng `ngram_range=(1, 2)` giúp học mối quan hệ của cả các từ đơn và cụm từ đi liền nhau (*"trúng thưởng"*, *"vay vốn"*).
4. **Phân loại & Suy luận:** Đưa ma trận số vào mô hình xác suất **Multinomial Naive Bayes** để tính toán phân lớp.

---

## 4. Hướng Dẫn Cài Đặt & Khởi Chạy

Mở Terminal (hoặc CMD / PowerShell) tại thư mục chứa dự án và thực hiện tuần tự các bước sau:

### Bước 1: Khởi tạo môi trường và cài đặt thư viện

```bash
# 1. Khởi tạo môi trường ảo độc lập
python -m venv .venv

# 2. Kích hoạt môi trường ảo (Dành cho Windows)
.venv\Scripts\activate

# 3. Cài đặt toàn bộ các thư viện bắt buộc
pip install -r requirements.txt

```

### Bước 2: Huấn luyện và Đóng gói hệ thống

Chạy script huấn luyện để đồng bộ tập dữ liệu Anh - Việt, in bảng so sánh hiệu năng 4 mô hình (Naive Bayes, Logistic Regression, SVM, Random Forest) và xuất file mô hình đóng gói:

```bash
python train.py

```

### Bước 3: Kiểm thử hệ thống

**Cách A: Kiểm thử trực tiếp trên Terminal**
Chạy lệnh sau để quét nhanh độ chính xác dựa trên bộ 10 test case hỗn hợp Anh - Việt:

```bash
python test.py

```

**Cách B: Sử dụng giao diện Web trực quan (Streamlit)**
Khởi chạy máy chủ giao diện Web bằng lệnh:

```bash
streamlit run app.py

```

Sau khi khởi chạy thành công, trình duyệt sẽ tự động mở ra giao diện Web:

* **Tab Nhập văn bản:** Cho phép gõ hoặc paste một câu tin nhắn bất kỳ để đo đồ thị phần trăm xác suất độ tin tưởng thời gian thực.
* **Tab Tải lên tệp tin:** Hỗ trợ upload file `.txt` hoặc `.csv` chứa danh sách tin nhắn để hệ thống tự động bóc tách từng dòng, xử lý hàng loạt và xuất ra bảng thống kê kết quả trực quan.

```

```