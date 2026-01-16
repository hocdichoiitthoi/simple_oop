import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from pathlib import Path

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
col_names = ['label', 'message']
data = pd.read_csv(url, sep='\t', names=col_names, header=None)
data['label_num'] = data.label.map({'ham':0, 'spam':1})

X = data['message'] 
y = data['label_num']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(X_train_raw), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test_raw), truncation=True, padding=True, max_length=128)

class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SMSDataset(train_encodings, list(y_train))
test_dataset = SMSDataset(test_encodings, list(y_test))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir=Path(__file__).parent / 'bert_results',
    num_train_epochs=2,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,
    warmup_steps=500,                
    weight_decay=0.01,              
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch"      
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("--- FINE-TUNING BERT ---")
trainer.train()

print("\n--- RESULTS BERT ---")
bert_results = trainer.evaluate()
print(bert_results)