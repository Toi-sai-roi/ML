Bản cũ (`README - old`) thì nội dung của nó đã **khớp khít 100% với đống code OLD** (hàm xử lý thô nằm ngoài Pipeline, có dùng Stemmer, chỉ chạy tiếng Anh) nên không cần phải sửa đổi gì về mặt logic hay tính năng của hệ thống cả.

Tuy nhiên, có một chỗ sai lệch thực tế nằm ở **Mục 4 (Kết quả đạt được)**. Trong file `train.py - old` của mày, tỷ lệ chia tập dữ liệu Train/Test thực tế đang cấu hình là **70/30** (`test_size=0.3`), nhưng trong file README cũ này lại đang viết nhầm thành **80/20**.

Đồng thời, để mày tiện lưu trữ hoặc đẩy cả bản cũ này lên GitHub (nếu muốn làm nhánh `old-version` hoặc lưu làm tài liệu đối chiếu), tao đã đính chính lại con số đó và đóng gói toàn bộ thành 1 bản `.md` hoàn chỉnh duy nhất. Mày chỉ cần copy đúng 1 lần dưới đây:

```markdown
# BÁO CÁO DỰ ÁN: HỆ THỐNG PHÂN LOẠI TIN NHẮN SPAM (SPAM CLASSIFIER) - PHIÊN BẢN NGUYÊN BẢN (OLD)

## 1. Giới thiệu
Đây là dự án ứng dụng Học máy (Machine Learning) và Xử lý ngôn ngữ tự nhiên (NLP) để tự động phân loại một đoạn văn bản (tin nhắn) là **Spam** (tin nhắn rác/quảng cáo/lừa đảo) hay **Ham** (tin nhắn bình thường).

Dự án sử dụng bộ dữ liệu kinhdefini **SMS Spam Collection** và thuật toán **Multinomial Naive Bayes** - một thuật toán hoạt động cực kỳ hiệu quả và nhanh chóng đối với các bài toán phân loại văn bản.

---

## 2. Cấu trúc thư mục dự án
Dự án bao gồm các file cốt lõi sau:
- **`requirements.txt`**: Danh sách các thư viện Python cần thiết để chạy dự án (pandas, scikit-learn, nltk, streamlit...).
- **`train.py`**: Script thực hiện toàn bộ quy trình: tự động tải dữ liệu, tiền xử lý thủ công, huấn luyện mô hình, đánh giá và lưu lại mô hình.
- **`app.py`**: Mã nguồn giao diện Web UI được xây dựng bằng Streamlit. File này load mô hình đã huấn luyện và cho phép người dùng nhập văn bản để test trực tiếp.
- **`test.py`**: Script kiểm thử nhanh mô hình trên Terminal mà không cần mở giao diện Web.
- **`spam_classifier.pkl`**: File chứa mô hình Machine Learning đã được huấn luyện xong (được tự động sinh ra sau khi chạy `train.py`).

---

## 3. Cách thức hoạt động (Quy trình xây dựng)

Hệ thống hoạt động dựa trên luồng xử lý 4 bước:

1. **Thu thập dữ liệu (Data Collection):** - Hệ thống tự động tải file ZIP từ kho dữ liệu máy học UCI, giải nén và nạp vào DataFrame qua `pandas`.
2. **Tiền xử lý văn bản (Text Preprocessing):** - *Làm sạch:* Chuyển toàn bộ văn bản về chữ thường, loại bỏ các dấu câu.
   - *Tokenize & Stopwords:* Tách câu thành các từ đơn lẻ, sau đó loại bỏ các "từ dừng" (như *the, is, in, at*... không có giá trị phân loại) sử dụng thư viện `nltk`.
   - *Stemming:* Đưa các từ biến thể về từ gốc (ví dụ: *running* -> *run*).
3. **Trích xuất đặc trưng (Feature Engineering):**
   - Sử dụng **TF-IDF** (Term Frequency-Inverse Document Frequency) để biến các từ văn bản thành các vector số học. Phương pháp này ưu tiên đánh giá cao các từ mang tính đặc trưng của Spam (như *winner, free, cash*) thay vì các từ xuất hiện phổ biến ở mọi nơi.
4. **Huấn luyện và Suy luận (Modeling):**
   - Đưa vector vào mô hình **Multinomial Naive Bayes** để học xác suất một tin nhắn rơi vào nhóm Spam hay Ham. 
   - Mô hình và TF-IDF sau đó được đóng gói thành một `Pipeline` duy nhất và lưu ra file `.pkl`. Khi dự đoán tin nhắn mới, văn bản chỉ cần đi qua đúng Pipeline này để ra kết quả.

---

## 4. Kết quả đạt được của mô hình
Sau quá trình huấn luyện với tỷ lệ chia tập dữ liệu Train/Test là 70/30, mô hình đạt được các chỉ số cực kỳ ấn tượng:
- **Độ chính xác tổng thể (Accuracy):** ~96.8%
- **F1-Score (dành cho Spam):** 0.87 (Đảm bảo bắt được đa số Spam mà không nhận diện nhầm tin nhắn quan trọng).
- **Precision (dành cho Spam):** 1.00 (Mức độ hoàn hảo: Tất cả các tin nhắn bị gắn nhãn Spam đều thực sự là Spam, không có ngoại lệ).

---

## 5. Hướng dẫn Cài đặt & Sử dụng

### Bước 1: Cài đặt thư viện
Mở Terminal (hoặc CMD/PowerShell) tại thư mục chứa dự án và chạy:
```bash
pip install -r requirements.txt

```

### Bước 2: Huấn luyện lại mô hình (Tùy chọn)

Nếu bạn muốn tự tay huấn luyện mô hình từ đầu, hãy chạy file `train.py`. Script sẽ tải dữ liệu và tiến hành học:

```bash
python train.py

```

### Bước 3: Kiểm tra thử mô hình (Testing)

Có 2 cách để kiểm tra mô hình:

**Cách A: Test nhanh qua Terminal (Không cần mở Web)**
Chạy lệnh sau để thấy kết quả dự đoán với 2 tin nhắn mẫu:

```bash
python test.py

```

**Cách B: Sử dụng qua giao diện Web (Streamlit)**
Chạy lệnh sau để khởi động máy chủ Web:

```bash
streamlit run app.py

```

*Lưu ý:* Khi lệnh chạy thành công, Terminal sẽ hiện ra 2 đường dẫn (Local URL và Network URL). Hãy copy một đường dẫn và dán vào trình duyệt Web của bạn để sử dụng giao diện. Tại đây, bạn có thể gõ bất kỳ tin nhắn nào và ấn "Kiểm tra" để xem kết quả trực quan!

```

```