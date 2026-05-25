# train.py - old
import os
import urllib.request
import zipfile
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Tải các gói tài nguyên cần thiết của thư viện NLTK để phục vụ tách từ và lọc từ dừng
nltk.download('stopwords', quiet=True)  # Danh sách từ dừng (ví dụ: is, am, are, the...)
nltk.download('punkt_tab', quiet=True)   # Bảng dữ liệu hỗ trợ thuật toán tách từ
nltk.download('punkt', quiet=True)       # Bộ tách từ (Tokenizer) mặc định của NLTK

# Định nghĩa các hằng số: Đường dẫn tải dữ liệu, tên file và tên model lưu trữ
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH = "smsspamcollection.zip"
DATA_FILE = "SMSSpamCollection"
MODEL_FILE = "spam_classifier.pkl"

def download_and_extract_data():
    """Hàm kiểm tra dữ liệu local. Nếu chưa có thì tự động tải từ UCI và giải nén"""
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)  # Tải file zip từ URL về máy
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")  # Giải nén file zip vào thư mục hiện tại
        os.remove(ZIP_PATH)          # Xóa file zip sau khi giải nén để dọn dẹp bộ nhớ
    else:
        print("Dataset already exists.")

def preprocess_text(text):
    """Hàm tiền xử lý ngôn ngữ tự nhiên (NLP) thủ công cho từng tin nhắn"""
    # 1. Chuyển đổi toàn bộ văn bản thành chữ thường để chuẩn hóa dữ liệu (ví dụ: 'Spam' và 'spam' là một)
    text = str(text).lower()
    
    # 2. Loại bỏ toàn bộ ký tự đặc biệt và dấu câu (string.punctuation chứa các ký tự như !, @, #, $, %,...)
    text = "".join([char for char in text if char not in string.punctuation])
    
    # 3. Tách chuỗi văn bản thô thành danh sách các từ đơn lẻ (Tokenization)
    tokens = nltk.word_tokenize(text)
    
    # 4. Loại bỏ từ dừng (Stopwords): Loại bỏ những từ xuất hiện nhiều nhưng không mang giá trị phân loại
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Đưa từ về gốc (Stemming) bằng thuật toán PorterStemmer (ví dụ: 'running', 'runs' -> 'run')
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # 6. Gộp danh sách các từ đã xử lý sạch sẽ ngược trở lại thành một chuỗi văn bản hoàn chỉnh
    return " ".join(tokens)

def main():
    # Bước 1: Tải và giải nén tập dữ liệu đầu vào
    download_and_extract_data()

    # Bước 2: Đọc dữ liệu vào DataFrame của Pandas. File phân tách bằng tab (\t), cấu trúc gồm 2 cột 'label' và 'message'
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['label', 'message'])
    print(f"Data shape: {df.shape}")  # In ra kích thước tập dữ liệu gốc (5572 dòng, 2 cột)

    # Bước 3: Áp dụng hàm tiền xử lý text cho toàn bộ các dòng tin nhắn trong cột 'message'
    print("Preprocessing text... (this may take a moment)")
    df['clean_message'] = df['message'].apply(preprocess_text)

    # Bước 4: Chia tập dữ liệu thành 2 phần Train (70%) và Test (30%) để huấn luyện và đánh giá độc lập
    # Tham số stratify=df['label'] cực kỳ quan trọng: Giúp giữ nguyên tỷ lệ mất cân bằng lớp (86.6% ham / 13.4% spam) trên cả tập train và test
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_message'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Cấu hình bộ trích xuất đặc trưng TF-IDF (Tập trung chuyển đổi văn bản thành ma trận số toán học)
    # max_features=3000: Chỉ giữ lại 3000 từ vựng có tần suất xuất hiện quan trọng nhất, loại bỏ các từ quá hiếm để giảm số chiều ma trận
    tfidf_params = {
        'max_features': 3000
    }

    # Định nghĩa cấu hình cấu trúc tham số cho 4 mô hình học máy cần đem ra thử nghiệm, so sánh
    # class_weight='balanced': Kỹ thuật gán trọng số lớp phạt nặng hơn cho lớp thiểu số (Spam), xử lý bài toán mất cân bằng dữ liệu gốc
    models = {
        "Naive Bayes":         MultinomialNB(),  # Thuật toán dựa trên xác suất Bayes, cực mạnh và nhẹ cho phân loại text văn bản
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),  # Mô hình phân loại tuyến tính xác suất
        "SVM":                 LinearSVC(class_weight='balanced', random_state=42),  # Tìm siêu phẳng tối ưu để phân tách 2 lớp dữ liệu
        "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),  # Mô hình Ensemble học máy dạng cây quyết định
    }

    # Mã hóa nhãn văn bản ('ham', 'spam') thành số (0, 1) phục vụ riêng cho việc tính toán số liệu đồ thị diện tích ROC AUC
    le = LabelEncoder()
    le.fit(y_train)
    y_test_enc = le.transform(y_test)

    print("\n" + "="*60)
    print("SO SANH CAC MO HINH (VERSION OLD)")
    print("="*60)

    # Vòng lặp chạy huấn luyện, kiểm thử và đo đạc kết quả chi tiết của từng mô hình một
    for name, clf in models.items():
        # Khởi tạo một Pipeline đóng gói chuỗi xử lý tuần tự: Trích xuất đặc trưng số TF-IDF -> Đưa vào bộ phân loại Clf
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', clf)
        ])
        
        # Tiến hành huấn luyện (Học máy) trên tập dữ liệu Train
        pipeline.fit(X_train, y_train)
        
        # Dự đoán nhãn (Ham/Spam) trên tập dữ liệu Test độc lập
        predictions = pipeline.predict(X_test)
        
        # Tính toán điểm số chính xác tổng thể (Accuracy Score)
        acc = accuracy_score(y_test, predictions)

        # Tính toán điểm ROC AUC (Đo lường khả năng phân biệt lớp của mô hình độc lập với ngưỡng xác suất)
        try:
            if hasattr(clf, "predict_proba"):
                scores = pipeline.predict_proba(X_test)[:, 1]  # Lấy xác suất của lớp Spam đối với các mô hình có hỗ trợ xác suất
            else:
                scores = pipeline.decision_function(X_test)   # Lấy khoảng cách tới siêu phẳng đối với mô hình tuyến tính SVM
            auc = roc_auc_score(y_test_enc, scores)
            auc_str = f"{auc:.4f}"
        except Exception:
            auc_str = "N/A"

        # In kết quả đánh giá chi tiết ra Terminal phục vụ cho việc lấy số liệu làm báo cáo Chương 4
        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"AUC      : {auc_str}")
        print(confusion_matrix(y_test, predictions))  # In ma trận nhầm lẫn biểu diễn các chỉ số TP, FP, TN, FN
        print(classification_report(y_test, predictions))  # In bảng chi tiết đo chỉ số Precision, Recall, F1-score của từng lớp dữ liệu

    print("="*60)
    
    # Bước 5: Lựa chọn mô hình Naive Bayes làm mô hình cốt lõi cuối cùng, huấn luyện lại độc lập và đóng gói lưu trữ xuống ổ cứng
    print(f"\nLuu mo hinh Naive Bayes vao {MODEL_FILE}...")
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('classifier', MultinomialNB())
    ])
    nb_pipeline.fit(X_train, y_train)
    
    # Dùng thư viện joblib để đóng băng đối tượng pipeline thành file nhị phân vật lý .pkl phục vụ tích hợp giao diện app.py
    joblib.dump(nb_pipeline, MODEL_FILE)
    print("Hoan tat!")

if __name__ == "__main__":
    main()