# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# 1. Tải dữ liệu mẫu (Iris dataset)
iris = load_iris()
X, y = iris.data, iris.target

# 2. Huấn luyện mô hình
clf = RandomForestClassifier()
clf.fit(X, y)

# 3. Lưu mô hình vào file
joblib.dump(clf, Path(__file__).parent / "model" / "model.pkl")
print("Đã lưu mô hình thành công! ")