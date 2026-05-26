# HỆ THỐNG PHÂN LOẠI TIN NHẮN SPAM (SPAM CLASSIFIER)

Dự án nghiên cứu, xây dựng và phát triển ứng dụng Học máy (Machine Learning) kết hợp Xử lý ngôn ngữ tự nhiên (NLP) để tự động nhận diện, phân loại tin nhắn rác (**Spam**) và tin nhắn thường (**Ham**).

Dự án được chia làm 2 giai đoạn phát triển rõ rệt, đại diện bởi 2 thư mục mã nguồn độc lập dưới đây:

---

## 📌 Cấu Trúc Toàn Bộ Dự Án

### 1. 📂 [Phiên Bản Nguyên Bản (OLD)](./old/hocmay-old/)
* **Đặc trưng:** Luồng xử lý dữ liệu thủ công nằm ngoài Pipeline. Sử dụng thuật toán cắt đuôi từ (*PorterStemmer*).
* **Ngôn ngữ hỗ trợ:** Chỉ phân loại tin nhắn bằng **Tiếng Anh** (Dựa trên bộ dữ liệu gốc SMS Spam Collection từ UCI).
* **Mô hình thử nghiệm:** Naive Bayes, Logistic Regression, SVM, Random Forest.
* 👉 [Xem chi tiết báo cáo và hướng dẫn bản OLD](./old/hocmay-old/README.md)

### 2. 📂 [Phiên Bản Cải Tiến Đa Ngôn Ngữ (NEW)](./new/hocmay-new/)
* **Đặc trưng:** Đóng gói trọn gói luồng tiền xử lý NLP vào thẳng `TfidfVectorizer` bên trong Scikit-learn Pipeline vật lý. Loại bỏ Stemming để tối ưu hóa ngôn ngữ.
* **Ngôn ngữ hỗ trợ:** Đa ngôn ngữ hỗn hợp **Anh - Việt** (Tích hợp thêm tập dữ liệu `spam_vietnamese.csv`).
* **Kỹ thuật nâng cao:** Xử lý bài toán mất cân bằng dữ liệu bằng kỹ thuật lấy mẫu (Down-sampling) tỷ lệ cân bằng gần 1:1.
* **Giao diện Web:** Tích hợp Streamlit UI cao cấp hỗ trợ nhập văn bản trực tiếp (đo xác suất % thời gian thực) và **Upload file để quét tin nhắn rác hàng loạt**.
* 👉 [Xem chi tiết báo cáo và hướng dẫn bản NEW](./new/hocmay-new/README.md)

---

## 🛠️ Hướng Dẫn Khởi Chạy Nhanh

Do dự án được chia làm 2 phân hệ độc lập, vui lòng di chuyển (`cd`) vào thư mục của phiên bản tương ứng trước khi thực hiện cài đặt môi trường và khởi chạy code.

### Ví dụ đối với Bản Cải Tiến (NEW):
```bash
# 1. Di chuyển vào thư mục bản mới
cd new/hocmay-new

# 2. Khởi tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate

# 3. Cài đặt thư viện và chạy huấn luyện mô hình
pip install -r requirements.txt
python train.py

# 4. Mở giao diện Web UI
streamlit run app.py

``` 
*Chi tiết các câu lệnh kiểm thử Terminal và cấu hình tham số sâu hơn, vui lòng truy cập vào file README riêng của từng thư mục.*
Dạ thì à... Bài toán của em đơn giản là phân loại tin nhắn spam, đầu vào là tệp tin lớn dạng tiếng Anh, đầu ra là các nhãn spam và ham ứng với từng tin. Model NB đã được train và lưu mô hình đó vào trong file ...pkl. trong bài này em còn dùng các mô hình khác để so sánh với nhau, ở đây em dùng SVM, Logic gì gì đó, forest gì gì đó... Và thấy được accuracy tổng của 4 cái thì thấy NB đang dẫn đầu với 100% tức là như nào... Tức là hoàn toàn gọi là model không bị nhầm lẫn giữa ham với spam gì hết khiến cho không có chuyện tin ham nào bị model gạt đi... Và còn nữa... Em có 1 vài những cải tiến đây ạ (tao chỉ vô 4.7 cùng với đó là show output train - new, sau đó là chỉ vô bảng 4.5 , dạ đây thưa cô,... đây là bản cải thiện bao gồm những thứ như là thêm dataset tiếng việt, loại bỏ stemming, phân loại để không trùng mấy tin ham/spam bị same... và trong output cũng thấy được thay đổi rõ rệt...  
