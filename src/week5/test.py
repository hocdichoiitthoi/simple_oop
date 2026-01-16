from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from pathlib import Path

model_path = Path(__file__).parent / "results_bert" / "checkpoint-558"

print("Loading model...")
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def predict_spam(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    return "SPAM" if predicted_class_id == 1 else "HAM"

test_msg_1 = "Urgent! You have won a 1 week free membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010"
test_msg_2 = "Hello, I'm going to be late for dinner today."

print(f"Message 1: {predict_spam(test_msg_1)}")
print(f"Message 2: {predict_spam(test_msg_2)}")