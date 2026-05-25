# train.py - new
import os
import urllib.request
import zipfile
import pandas as pd
import nltk
from nltk.corpus import stopwords
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

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH = "smsspamcollection.zip"
ENGLISH_DATA_FILE = "SMSSpamCollection"
VIETNAMESE_DATA_FILE = "spam_vietnamese.csv" 
MODEL_FILE = "spam_classifier.pkl"

def download_and_extract_data():
    if not os.path.exists(ENGLISH_DATA_FILE):
        print("Đang tải dataset tiếng Anh từ UCI...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(ZIP_PATH)
    else:
        print("Dataset tiếng Anh đã tồn tại.")

def preprocess_text(text):
    """Hàm tiền xử lý tùy chỉnh tích hợp thẳng vào Pipeline"""
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    tokens = [word for word in tokens if word not in stop_words]
    # Đã loại bỏ PorterStemmer để không làm méo từ Tiếng Việt theo mục 4.7 & Chương 5
    return " ".join(tokens)

def main():
    # 1. Lấy và chuẩn bị dữ liệu tiếng Anh
    download_and_extract_data()
    
        # Đọc dữ liệu gốc với 2 cột: nhãn (label) và nội dung tin nhắn (message)
    df_en = pd.read_csv(ENGLISH_DATA_FILE, sep='\t', header=None, names=['label', 'message'])
    
        # Loại bỏ các tin nhắn trùng lặp để tránh mô hình bị học vẹt (overfitting)
    df_en = df_en.drop_duplicates(subset=['message'])
    
        # Tách dữ liệu thành 2 tập riêng biệt theo nhãn spam và ham
    spam_df = df_en[df_en['label'] == 'spam']
    ham_df = df_en[df_en['label'] == 'ham']
    
        # Xác định số lượng mẫu cần lấy dựa trên số lượng ít nhất của 2 nhóm (giới hạn trần 800)
    n = min(len(spam_df), len(ham_df), 800)
    
        # Trích xuất ngẫu nhiên số lượng mẫu bằng nhau (n mẫu) từ mỗi nhóm để cân bằng dữ liệu 1:1
    df_en = pd.concat([
        spam_df.sample(n, random_state=42),
        ham_df.sample(n, random_state=42)
    ], ignore_index=True)
    
        # In kiểm tra số lượng phân bổ thực tế của các nhãn sau khi cân bằng
    print(f"After sampling English: {df_en['label'].value_counts().to_dict()}")
    
    # 2. Lấy dữ liệu tiếng Việt từ file CSV 
    if os.path.exists(VIETNAMESE_DATA_FILE):
        print(f"Đang load dữ liệu tiếng Việt từ {VIETNAMESE_DATA_FILE}...")
        df_vi = pd.read_csv(VIETNAMESE_DATA_FILE)
        df_vi.columns = ['label', 'message']
    else:
        print(f"LỖI: Không tìm thấy file {VIETNAMESE_DATA_FILE}. Cần tạo file này trước!")
        return

    # 3. Gộp dữ liệu thô (Để Pipeline tự xử lý tiền xử lý, tránh lặp dữ liệu)
    print("Đang trộn dữ liệu Anh - Việt...")
    df = pd.concat([df_en, df_vi], ignore_index=True)
    df = df.drop_duplicates(subset=['message'])

    print(f"Tổng số mẫu dữ liệu: {df.shape[0]}")
    print(f"Phân bổ: \n{df['label'].value_counts()}")

    # 4. Chia train/test dựa trên tin nhắn GỐC (Raw Message)
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    # 5. So sánh các mô hình bằng cách đẩy hàm xử lý vào preprocessor của Pipeline
    # Thiết lập stop_words=None trong Tfidf để không ghi đè hàm xử lý tiếng Việt thủ công
    tfidf_params = {
        'max_features': 3000,
        'lowercase': False, 
        'preprocessor': preprocess_text,
        'ngram_range': (1, 2)
    }
    
    models = {
        "Naive Bayes":         MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "SVM":                 LinearSVC(class_weight='balanced', random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    }

    le = LabelEncoder()
    le.fit(y_train)
    y_test_enc = le.transform(y_test)

    print("\n" + "="*60)
    print("SO SANH CAC MO HINH (VERSION NEW - ANH VIET PIPELINE)")
    print("="*60)

    for name, clf in models.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(**tfidf_params)),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        try:
            if hasattr(clf, "predict_proba"):
                scores = pipeline.predict_proba(X_test)[:, 1]
            else:
                scores = pipeline.decision_function(X_test)
            auc = roc_auc_score(y_test_enc, scores)
            auc_str = f"{auc:.4f}"
        except Exception:
            auc_str = "N/A"

        print(f"\n--- {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"AUC      : {auc_str}")
        print(confusion_matrix(y_test, predictions))
        print(classification_report(y_test, predictions))

    # 6. Huấn luyện và Lưu model Naive Bayes tối ưu nhất
    print("="*60)
    print(f"\nLưu mô hình Naive Bayes vào {MODEL_FILE}...")
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('classifier', MultinomialNB())
    ])
    nb_pipeline.fit(X_train, y_train)
    joblib.dump(nb_pipeline, MODEL_FILE)
    print("Hoàn tất huấn luyện bản nâng cấp!")

if __name__ == "__main__":
    main()