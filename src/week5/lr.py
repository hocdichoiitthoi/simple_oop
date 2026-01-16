import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from wordcloud import WordCloud
nltk.download('stopwords')

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
col_names = ['label', 'message']
data = pd.read_csv(url, sep='\t', names=col_names, header=None)

stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))

def clean_text_regex(text):
    """
    Lowercase, Regex substitutions, Remove punctuation
    """
    text = text.lower()   
    text = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
    text = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', text)
    text = re.sub(r'[£$€]', 'moneysymb', text)
    text = re.sub(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})(?: *x(\d+))?\b', 'phonenum', text)
    text = re.sub(r'\d+(\.\d+)?', 'numbr', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def advanced_preprocess(text):
    """
    Spell Correction, Stemming, Stopwords removal
    """
    tokens = text.split()
    clean_tokens = []
    
    for token in tokens:
        if token in stop_words:
            continue
        token = stemmer.stem(token)  
        clean_tokens.append(token)
        
    return " ".join(clean_tokens)

data['label_num'] = data.label.map({'ham':0, 'spam':1})
df_clean = data.copy()

print("Đang xử lý Regex (Bước 1-3)...")
df_clean['clean_msg'] = df_clean['message'].apply(clean_text_regex)

print("Đang xử lý NLP nâng cao (Bước 4-6)... Điều này có thể mất vài phút.")

df_clean['clean_msg'] = df_clean['clean_msg'].apply(advanced_preprocess)

print("\n--- SO SÁNH TRƯỚC VÀ SAU ---")
print(df_clean[['message', 'clean_msg']].head())

X_train, X_test, y_train, y_test = train_test_split(
    df_clean['clean_msg'], 
    df_clean['label_num'],
    test_size=0.2, 
    random_state=42
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

print("\n--- Training với TF-IDF Vectorizer ---")
vect_tfidf = TfidfVectorizer() 
X_train_tfidf = vect_tfidf.fit_transform(X_train)
X_test_tfidf = vect_tfidf.transform(X_test)

clf_tfidf = LogisticRegression(solver='liblinear', class_weight='balanced')
clf_tfidf.fit(X_train_tfidf, y_train)
print(classification_report(y_test, clf_tfidf.predict(X_test_tfidf), target_names=['Ham', 'Spam']))

feature_names = vect_tfidf.get_feature_names_out()
coefficients = clf_tfidf.coef_.flatten()

df_features = pd.DataFrame({'word': feature_names, 'coef': coefficients})

top_spam_features = df_features.sort_values(by='coef', ascending=False).head(5)
top_ham_features = df_features.sort_values(by='coef', ascending=True).head(5)

print("--- TOP 5 TỪ KHÓA ĐẶC TRƯNG CHO SPAM ---")
print(top_spam_features)

print("\n--- TOP 5 TỪ KHÓA ĐẶC TRƯNG CHO HAM ---")
print(top_ham_features)

spam_text = " ".join(df_clean[df_clean['label_num'] == 1]['clean_msg'])
ham_text = " ".join(df_clean[df_clean['label_num'] == 0]['clean_msg'])

wc_spam = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(spam_text)
wc_ham = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(ham_text)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(wc_spam, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - SPAM (Tin rác)', fontsize=16)

plt.subplot(1, 2, 2)
plt.imshow(wc_ham, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - HAM (Tin thường)', fontsize=16)

output_folder = Path(__file__).parent / 'visualize'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"File save: {output_folder}")
save_path = os.path.join(output_folder, 'spam_wordcloud.png')
plt.savefig(save_path)