# app.py - new
import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd

MODEL_FILE = "spam_classifier.pkl"

# BẮT BUỘC PHẢI KHAI BÁO HÀM NÀY LÊN ĐẦU ĐỂ JOBLIB ĐỐI CHIẾU KHI LOAD MODEL
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
    return " ".join(tokens)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

st.title("Phân loại tin nhắn Spam (Spam Classifier) - Bản Cải Tiến")
st.write("Nhập tin nhắn hoặc tải lên tệp tin để kiểm tra xem nó là Spam hay Tin nhắn thường (Ham).")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Không tìm thấy file mô hình `spam_classifier.pkl`. Vui lòng chạy `train.py` trước.")
    st.stop()

# Layout các tab nhập dữ liệu đầu vào theo mục 4.7 của báo cáo
tab1, tab2 = st.tabs(["Nhập văn bản", "Tải lên tệp tin"])

with tab1:
    user_input = st.text_area("Nội dung tin nhắn:", height=150)

with tab2:
    uploaded_file = st.file_uploader("Chọn tệp tin văn bản (.txt, .md, .csv) để kiểm tra:", type=["txt", "md", "csv"])

if st.button("Kiểm tra"):
    messages_to_check = []
    
    # 1. Thu thập dữ liệu từ File Upload nếu có
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'message' in df.columns:
                    messages_to_check = df['message'].dropna().astype(str).tolist()
                else:
                    st.error("File CSV phải có cột tên là 'message'!")
                    st.stop()
            else:
                # Đối với .txt hoặc .md: Tách thành từng dòng tin nhắn riêng biệt
                content = uploaded_file.read().decode("utf-8")
                messages_to_check = [line.strip() for line in content.split("\n") if line.strip()]
                
            st.info(f"Đã đọc thành công {len(messages_to_check)} tin nhắn từ tệp tin.")
        except Exception as e:
            st.error(f"Lỗi khi đọc file: {e}")
            st.stop()
            
    # 2. Thu thập dữ liệu từ Text Area (Nếu không upload file)
    elif user_input.strip() != "":
        messages_to_check = [user_input.strip()]

    # 3. Tiến hành kiểm tra và hiển thị kết quả
    if not messages_to_check:
        st.warning("Vui lòng nhập tin nhắn hoặc tải lên tệp tin cần kiểm tra!")
    else:
        # TRƯỜNG HỢP 1: Chỉ có 1 tin nhắn (Hiện giao diện progress bar đo % của mày)
        if len(messages_to_check) == 1:
            msg = messages_to_check[0]
            prediction = model.predict([msg])[0]
            proba = model.predict_proba([msg])[0]
            
            classes = model.classes_
            prob_dict = dict(zip(classes, proba))
            spam_pct = prob_dict.get("spam", 0)
            ham_pct = prob_dict.get("ham", 0)

            if prediction == "spam":
                st.error("🚨 CẢNH BÁO: Đây có thể là nội dung SPAM!")
            else:
                st.success("✅ Đây có vẻ là nội dung bình thường (HAM).")

            st.markdown("**Độ tin tưởng của mô hình:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ HAM", f"{ham_pct:.1%}")
                st.progress(ham_pct)
            with col2:
                st.metric("🚨 SPAM", f"{spam_pct:.1%}")
                st.progress(spam_pct)
                
        # TRƯỜNG HỢP 2: Kiểm tra hàng loạt từ File (Hiện ra bảng kết quả tổng quan)
        else:
            results = []
            for msg in messages_to_check:
                pred = model.predict([msg])[0]
                proba = model.predict_proba([msg])[0]
                prob_dict = dict(zip(model.classes_, proba))
                
                results.append({
                    "Nội dung tin nhắn": msg,
                    "Kết quả": pred.upper(),
                    "Xác suất Spam": f"{prob_dict.get('spam', 0):.1%}"
                })
            
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True)
            
            # Hiện số lượng thống kê nhanh
            spam_cnt = (df_res["Kết quả"] == "SPAM").sum()
            st.metric("Tổng số tin nhắn rác (SPAM) phát hiện:", f"{spam_cnt} / {len(messages_to_check)}")