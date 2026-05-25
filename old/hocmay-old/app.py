# app.py - old
import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

MODEL_FILE = "spam_classifier.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

st.title("Phân loại tin nhắn Spam (Spam Classifier)")
st.write("Nhập tin nhắn vào bên dưới để kiểm tra xem nó là Spam hay Tin nhắn thường (Ham).")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Không tìm thấy file mô hình `spam_classifier.pkl`. Vui lòng chạy `train.py` trước.")
    st.stop()

user_input = st.text_area("Nội dung tin nhắn:", height=150)

if st.button("Kiểm tra"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập tin nhắn cần kiểm tra!")
    else:
        clean_input = preprocess_text(user_input)
        prediction = model.predict([clean_input])[0]
        proba = model.predict_proba([clean_input])[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, proba))
        spam_pct = prob_dict.get("spam", 0)
        ham_pct = prob_dict.get("ham", 0)

        if prediction == "spam":
            st.error("🚨 CẢNH BÁO: Đây có thể là tin nhắn SPAM!")
        else:
            st.success("✅ Đây có vẻ là tin nhắn bình thường (HAM).")

        st.markdown("**Độ tin tưởng của mô hình:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ HAM", f"{ham_pct:.1%}")
            st.progress(ham_pct)
        with col2:
            st.metric("🚨 SPAM", f"{spam_pct:.1%}")
            st.progress(spam_pct)