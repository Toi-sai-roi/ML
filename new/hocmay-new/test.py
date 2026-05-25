# test.py - new
import joblib
import nltk
from nltk.corpus import stopwords
import string

# BẮT BUỘC PHẢI CÓ HÀM NÀY ĐỂ JOBLIB ĐỐI CHIẾU KHI TẢI MODEL
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

if __name__ == "__main__":
    try:
        model = joblib.load("spam_classifier.pkl")
    except FileNotFoundError:
        print("Model file not found! Vui lòng chạy train.py - new trước để sinh file mô hình.")
        exit()

    test_cases = [
        # Tiếng Anh
        ("WINNER! You have been selected to receive a $1000 cash prize. Click here to claim your reward.", "SPAM"),
        ("Hey, are we still meeting for lunch tomorrow at 12?", "HAM"),
        
        # Tiếng Việt Spam
        ("Chúc mừng bạn đã trúng thưởng phần quà trị giá 5 triệu đồng. Click ngay vào link để nhận thưởng.", "SPAM"),
        ("Nhận việc làm tại nhà lương 500k-1tr/ngày chỉ với điện thoại. Liên hệ Zalo 0987654321", "SPAM"),
        ("Tài khoản của bạn đã bị khóa. Vui lòng đăng nhập lại tại link abc.xyz để mở khóa.", "SPAM"),
        ("Hỗ trợ vay vốn ngân hàng lãi suất 0%, không cần thế chấp. Lấy tiền trong 10 phút.", "SPAM"),
        
        # Tiếng Việt Ham
        ("Tối nay mấy giờ cậu đi làm về? Nhớ mua đồ ăn tối nhé.", "HAM"),
        ("Anh ơi em gửi lại file báo cáo tài chính, anh kiểm tra giúp em với.", "HAM"),
        ("Mai chủ nhật rảnh không đi đá bóng với bọn tớ nhé.", "HAM"),
        ("Sếp nhắc chiều nay họp lúc 3h ở phòng họp lớn nhé mọi người.", "HAM")
    ]
    
    print("Testing messages with New Pipeline...")
    print(f"\n{'Tin nhắn':<55} {'Thực tế':<8} {'Dự đoán':<8} {'Spam%'}")
    print("-" * 85)
    
    for msg, expected in test_cases:
        pred = model.predict([msg])[0]
        prob = model.predict_proba([msg])[0]
        
        classes = list(model.classes_)
        spam_pct = prob[classes.index('spam')] * 100
        
        status = "✅" if pred.upper() == expected else "❌"
        
        # msg[:54] cắt chuỗi hiển thị gọn gàng trên 1 dòng ở terminal
        print(f"{msg[:54]:<55} {expected:<8} {pred.upper():<6}{status}  {spam_pct:.1f}%")