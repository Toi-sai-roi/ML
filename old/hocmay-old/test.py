# test.py - old
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

model = joblib.load("spam_classifier.pkl")

test_msgs = [
    ("Congratulations! You've won a FREE iPhone! Click here to claim now!", "SPAM"),
    ("URGENT: Your bank account has been compromised. Verify immediately!", "SPAM"),
    ("Win £1000 cash! Text WIN to 12345 now!", "SPAM"),
    ("Hey, are you coming to the party tonight?", "HAM"),
    ("I'll call you when I get out of this meeting", "HAM"),
    ("Can you pick up some groceries on your way home?", "HAM"),
]

print(f"{'Tin nhắn':<55} {'Thực tế':<8} {'Dự đoán':<8} {'Spam%'}")
print("-" * 85)
for msg, expected in test_msgs:
    clean = preprocess_text(msg)
    pred  = model.predict([clean])[0]
    prob  = model.predict_proba([clean])[0]
    classes = list(model.classes_)
    spam_pct = prob[classes.index('spam')] * 100
    status = "✅" if pred.upper() == expected else "❌"
    print(f"{msg[:54]:<55} {expected:<8} {pred.upper():<6}{status}  {spam_pct:.1f}%")