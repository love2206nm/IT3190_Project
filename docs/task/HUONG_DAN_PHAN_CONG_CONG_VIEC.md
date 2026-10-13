# 📋 BẢN HƯỚNG DẪN KIẾN TRÚC & PHÂN CÔNG CÔNG VIỆC BÀI TẬP LỚN
**Môn học:** Nhập môn Học máy và Khai phá dữ liệu (IT3190)  
**Đề tài:** Phân loại cảm xúc văn bản tiếng Việt (UIT-VSMEC) kết hợp LLM phản hồi thông minh  
**Nhóm trưởng:** Hải Nam  

---

## 📌 I. TỔNG QUAN KIẾN TRÚC DỰ ÁN & BẢN ĐỒ KHỐI CHỨC NĂNG

### 1. Kiến trúc phân tầng và Luồng dữ liệu (Architecture & Data Flow)
Hệ thống được thiết kế theo kiến trúc phân tầng dạng đường ống (Pipeline Architecture). Mỗi thành viên làm chủ một **Khối chức năng (Subsystem)** độc lập:

```
[Raw Text Input]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. KHỐI TIỀN XỬ LÝ DỮ LIỆU (Phụ trách: Mai Hân)        │
│    • Module: source/data_process/preprocessing/        │
│    • Output: Text sạch chuẩn hóa                       │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ 2. KHỐI TRÍCH XUẤT ĐẶC TRƯNG & LINEAR ML (Khôi)        │
│    • Module: featuring/ & classifier/algorithms/linear/│
│    • Output: TF-IDF Vector & Nhóm mô hình Linear / SVM │
└───────────────────────┬────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
┌───────────────────────────┐ ┌──────────────────────────┐
│ 3. KHỐI CÂY QUYẾT ĐỊNH &  │ │ 4. KHỐI ĐÁNH GIÁ, SO SÁNH│
│    ENSEMBLE ML (Kiên)     │ │    & PHÂN TÍCH LỖI (Anh) │
│    • Module: .../tree/    │ │    • Module: evaluator.py│
│    • Output: DecisionTree │ │    • Output: Benchmark,  │
│      & RandomForest       │ │      Confusion Matrix    │
└──────────────┬────────────┘ └────────────┬─────────────┘
               │                           │
               └──────────────┬────────────┘
                              ▼
┌────────────────────────────────────────────────────────┐
│ 5. KHỐI ĐIỀU PHỐI, TÍCH HỢP LLM & GIAO DIỆN (Hải Nam)  │
│    • Module: services/, api/, ui/, model/llm/          │
│    • Output: FastAPI Server, LLM Gemini, Streamlit Web │
└────────────────────────────────────────────────────────┘
```

---

## 👥 II. BẢNG PHÂN CÔNG KHỐI CHỨC NĂNG HỆ THỐNG

Thay vì giao các dòng lệnh đơn lẻ, mỗi thành viên **làm chủ hoàn toàn 1 khối chức năng** trong hệ thống:

| Thành viên & MSSV | Khối chức năng phụ trách (Subsystem) | Thư mục sở hữu độc quyền | Trách nhiệm đầu ra của khối (Deliverables) | Nhánh Git |
| :--- | :--- | :--- | :--- | :--- |
| **Bùi Mai Hân**<br>`202400041` | **Khối Tiền xử lý dữ liệu**<br>*(Data Preprocessing Subsystem)* | `source/data_process/preprocessing/`<br>`data/clean_data/`<br>`notebooks/01_...` | • Class `TextCleaner`<br>• Từ điển `teencode_dict.py`<br>• Dữ liệu sạch `clean_data/*.json`<br>• Biểu đồ phân bố nhãn `docs/figures/` | `feature/data-preprocessing` |
| **Vũ Đình Khôi**<br>`20235124` | **Khối Trích xuất Đặc trưng & Mô hình Tuyến tính**<br>*(Feature Engineering & Linear ML)* | `source/data_process/featuring/`<br>`source/model/classifier/base_classifier.py`<br>`source/model/classifier/algorithms/linear_models.py`<br>`notebooks/02_...` | • Class `TfidfFeatureExtractor`<br>• Lớp trừu tượng `BaseSentimentClassifier`<br>• 3 mô hình: Naive Bayes, LogReg, LinearSVC<br>• File lưu: `best_model.joblib`, `tfidf.joblib` | `feature/ml-linear-models` |
| **Vũ Trung Kiên**<br>`202416253` | **Khối Thực nghiệm Mô hình Cây & Ensemble**<br>*(Tree-based & Ensemble Subsystem)* | `source/model/classifier/algorithms/tree_models.py`<br>`notebooks/03_...` | • Class `DecisionTreeModel`<br>• Class `RandomForestModel`<br>• Notebook thực nghiệm so sánh mô hình cây trên ma trận thưa TF-IDF | `feature/ml-tree-models` |
| **Nguyễn Hoàng Anh**<br>`202416121` | **Khối Đánh giá & Phân tích Chất lượng Mô hình**<br>*(Evaluation & Quality Analytics)* | `source/model/classifier/evaluator.py`<br>`notebooks/04_...`<br>`docs/figures/` | • Class `ModelEvaluator`<br>• Bảng so sánh Benchmark đối đầu 5 mô hình<br>• Ảnh biểu đồ Heatmap Confusion Matrix<br>• Báo cáo phân tích 5 câu đoán sai (Error Analysis) | `feature/model-evaluation` |
| **Hải Nam**<br>*(Nhóm trưởng)* | **Khối Điều phối Dịch vụ, LLM & Ứng dụng Web**<br>*(Core Orchestration & Full App)* | `source/services/`<br>`source/model/llm/`<br>`source/api/`<br>`source/ui/`<br>`deploy/`, `docs/report/` | • Tầng điều phối `OrchestratorService`<br>• Module kết nối Google Gemini sinh phản hồi CSKH<br>• REST API (FastAPI) & Web Demo (Streamlit)<br>• Docker Compose, Báo cáo & Slide tổng thể | `main` |

---

## 🛡️ III. NGUYÊN TẮC THIẾT KẾ & BẢO VỆ HỆ THỐNG (CHỐNG CONFLICT)

1. **Ranh giới thư mục bất khả xâm phạm:** Mỗi thành viên chỉ thao tác trong thư mục sở hữu của mình. Không tạo file ở thư mục gốc, không sửa file của người khác.
2. **Tuân thủ Hợp đồng Giao tiếp (Communication Contracts):**
   * Output của Mai Hân (`clean_text`) là Input của Đình Khôi (`transform`).
   * Output của Đình Khôi (`tfidf_matrix`) là Input của Trung Kiên (`tree_models`).
   * Output của Khôi & Kiên (`predict`) là Input của Hoàng Anh (`evaluator`).
3. **Cấm tuyệt đối đường dẫn tuyệt đối:** Sử dụng `pathlib.Path` để lấy đường dẫn tương đối từ thư mục gốc.

---

## 📖 IV. HƯỚNG DẪN CHI TIẾT TỪNG BƯỚC CHO TỪNG THÀNH VIÊN

---

### 1. Bùi Mai Hân — Khối Tiền xử lý dữ liệu (Data Preprocessing)
* **Vai trò trong dự án:** Cổng gác đầu tiên làm sạch văn bản mạng xã hội nhiều teencode, icon, spam link.
* **Input nhận vào:** `data/raw_data/*.json` $\rightarrow$ **Output bàn giao:** `data/clean_data/*.json` & class `TextCleaner`.

#### Các bước thực hiện chi tiết:
* **Bước 1 (Tạo từ điển teencode):** Tạo file `source/data_process/preprocessing/teencode_dict.py`:
  ```python
  TEENCODE_DICT = {
      "ko": "không", "k": "không", "khum": "không",
      "dc": "được", "đc": "được", "vs": "với",
      "t": "tôi", "mik": "mình", "bt": "biết", "lm": "làm"
  }
  ```
* **Bước 2 (Cài đặt Class `TextCleaner`):** Tạo file `source/data_process/preprocessing/text_cleaner.py`:
  ```python
  import re, json
  from pathlib import Path
  from source.data_process.preprocessing.teencode_dict import TEENCODE_DICT

  class TextCleaner:
      def __init__(self, teencode_dict: dict = None):
          self.teencode = teencode_dict or TEENCODE_DICT

      def clean_text(self, text: str) -> str:
          if not isinstance(text, str): return ""
          text = text.lower()                                 # 1. Chữ thường
          text = re.sub(r'https?://\S+|www\.\S+', '', text)   # 2. Xóa URL
          text = re.sub(r'[^\w\s]', ' ', text)               # 3. Xóa ký tự đặc biệt thừa
          words = [self.teencode.get(w, w) for w in text.split()] # 4. Thay teencode
          return ' '.join(words).strip()                      # 5. Xóa khoảng trắng thừa

      def clean_dataset(self, input_path: Path, output_path: Path):
          with open(input_path, 'r', encoding='utf-8') as f: data = json.load(f)
          for item in data: item["Sentence"] = self.clean_text(item["Sentence"])
          with open(output_path, 'w', encoding='utf-8') as f:
              json.dump(data, f, ensure_ascii=False, indent=2)
  ```
* **Bước 3 (Tự kiểm thử & Chạy xuất dữ liệu):** Viết `if __name__ == "__main__":` test thử 1 câu. Sau đó gọi `clean_dataset` xuất 3 file sạch vào `data/clean_data/`.
* **Bước 4 (Khai phá EDA & Bàn giao):** Mở `notebooks/01_eda_and_visualizations.ipynb`, vẽ biểu đồ cột phân bố 7 nhãn lưu vào `docs/figures/eda_label_distribution.png`. Báo cho Khôi và đẩy code lên nhánh `feature/data-preprocessing`.

---

### 2. Vũ Đình Khôi — Khối Đặc trưng & Mô hình Tuyến tính (Linear ML)
* **Vai trò trong dự án:** Xây dựng nền tảng thuật toán (Core ML), chuyển hóa văn bản thành vector TF-IDF và huấn luyện các mô hình tuyến tính (Naive Bayes, SVM, LogReg). Định nghĩa Base Class cho cả nhóm.
* **Input nhận vào:** Dữ liệu sạch từ Hân $\rightarrow$ **Output bàn giao:** `tfidf_vectorizer.joblib`, `best_model.joblib`, Base Interface cho Kiên.

#### Các bước thực hiện chi tiết:
* **Bước 1 (Định nghĩa Lớp trừu tượng):** Tạo file `source/model/classifier/base_classifier.py`:
  ```python
  from abc import ABC, abstractmethod
  import numpy as np

  class BaseSentimentClassifier(ABC):
      @abstractmethod
      def train(self, X_train, y_train): pass
      @abstractmethod
      def predict(self, X_test) -> np.ndarray: pass
      @abstractmethod
      def predict_proba(self, X_test) -> np.ndarray: pass # % xác suất cho Web UI
      @abstractmethod
      def save(self, filepath: str): pass
  ```
* **Bước 2 (Xây dựng Class TF-IDF):** Tạo file `source/data_process/featuring/tfidf_extractor.py`:
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  import joblib

  class TfidfFeatureExtractor:
      def __init__(self, max_features=5000, ngram_range=(1, 2)):
          self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
      def fit_transform(self, texts): return self.vectorizer.fit_transform(texts)
      def transform(self, texts): return self.vectorizer.transform(texts)
      def save(self, filepath): joblib.dump(self.vectorizer, filepath)
  ```
* **Bước 3 (Cài đặt 3 mô hình tuyến tính):** Trong `source/model/classifier/algorithms/linear_models.py`, kế thừa `BaseSentimentClassifier` để viết 3 class: `MultinomialNBClassifier`, `LogisticRegressionClassifier`, `LinearSVCClassifier` (dùng `CalibratedClassifierCV(LinearSVC())` để có xác suất).
* **Bước 4 (Huấn luyện, Lưu model & Bàn giao):** Chạy `notebooks/02_linear_models.ipynb`, lưu model F1 cao nhất vào `source/model/classifier/saved_models/best_model.joblib`. Cung cấp vectorizer cho Kiên và số liệu cho Hoàng Anh.

---

### 3. Vũ Trung Kiên — Khối Mô hình Cây & Ensemble (Tree ML)
* **Vai trò trong dự án:** Thực nghiệm nhánh mô hình phi tuyến (Decision Tree, Random Forest) trên ma trận thưa TF-IDF để hoàn thiện so sánh đối đầu trong báo cáo.
* **Input nhận vào:** TF-IDF từ Khôi $\rightarrow$ **Output bàn giao:** File `tree_models.py` và bảng thời gian/F1.

#### Các bước thực hiện chi tiết:
* **Bước 1 (Cài đặt Class mô hình Cây):** Tạo file `source/model/classifier/algorithms/tree_models.py`:
  ```python
  from source.model.classifier.base_classifier import BaseSentimentClassifier
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  import joblib

  class DecisionTreeModel(BaseSentimentClassifier):
      def __init__(self, max_depth=20, random_state=42):
          self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
      def train(self, X_train, y_train): self.model.fit(X_train, y_train)
      def predict(self, X_test): return self.model.predict(X_test)
      def predict_proba(self, X_test): return self.model.predict_proba(X_test)
      def save(self, filepath): joblib.dump(self.model, filepath)

  class RandomForestModel(BaseSentimentClassifier):
      def __init__(self, n_estimators=100, max_depth=20, random_state=42):
          self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
      def train(self, X_train, y_train): self.model.fit(X_train, y_train)
      def predict(self, X_test): return self.model.predict(X_test)
      def predict_proba(self, X_test): return self.model.predict_proba(X_test)
      def save(self, filepath): joblib.dump(self.model, filepath)
  ```
* **Bước 2 (Chạy thực nghiệm & Đo thời gian):** Mở `notebooks/03_tree_models.ipynb`, nạp TF-IDF từ Khôi, huấn luyện 2 mô hình. Dùng `time.time()` đo thời gian huấn luyện của Random Forest so với LinearSVC.
* **Bước 3 (Rút ra kết luận học thuật & Bàn giao):** Ghi nhận Accuracy, Macro-F1, Train Time gửi qua Hoàng Anh; phân tích vì sao Cây quyết định dễ bị overfit trên ma trận thưa TF-IDF.

---

### 4. Nguyễn Hoàng Anh — Khối Đánh giá & Phân tích Chất lượng (Evaluation Analytics)
* **Vai trò trong dự án:** Kiểm định chất lượng độc lập, tổng hợp so sánh 5 mô hình, vẽ ma trận nhầm lẫn và mổ xẻ nguyên nhân đoán sai.
* **Input nhận vào:** Dự đoán `y_true`, `y_pred` của 5 mô hình từ Khôi & Kiên $\rightarrow$ **Output bàn giao:** Benchmark Table, Confusion Matrix, 5 câu phân tích lỗi cho Slide.

#### Các bước thực hiện chi tiết:
* **Bước 1 (Cài đặt Class `ModelEvaluator`):** Tạo file `source/model/classifier/evaluator.py`:
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

  class ModelEvaluator:
      @staticmethod
      def compute_metrics(y_true, y_pred) -> dict:
          return {"accuracy": round(accuracy_score(y_true, y_pred), 4), "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4)}

      @staticmethod
      def plot_and_save_confusion_matrix(y_true, y_pred, labels: list, save_path: str):
          cm = confusion_matrix(y_true, y_pred, labels=labels)
          plt.figure(figsize=(8, 6))
          sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
          plt.title('Ma Trận Nhầm Lẫn (Confusion Matrix)')
          plt.xlabel('Dự Đoán'); plt.ylabel('Thực Tế')
          plt.tight_layout()
          plt.savefig(save_path, dpi=300); plt.close()
  ```
* **Bước 2 (Lập bảng Benchmark đối đầu 5 mô hình):** Mở `notebooks/04_benchmarking_and_errors.ipynb`, tổng hợp kết quả của Khôi & Kiên lập bảng so sánh chuẩn: Naive Bayes, LogReg, LinearSVC, Decision Tree, Random Forest.
* **Bước 3 (Vẽ Confusion Matrix & Phân tích lỗi):** Chạy xuất ảnh `docs/figures/confusion_matrix_best.png`. Lọc 3-5 câu mô hình đoán sai (câu mỉa mai, khen đểu, teencode lạ) giải thích lý do để đưa vào Slide thuyết trình.

---

### 5. Hải Nam (Nhóm trưởng) — Khối Điều phối, LLM & Web App
* **Nhiệm vụ:**
  * **Khối LLM (`source/model/llm/`):** Tích hợp Google Gemini sinh phản hồi CSKH xoa dịu/tri ân theo nhãn cảm xúc.
  * **Khối Điều phối (`source/services/orchestrator_service.py`):** Nối luồng `TextCleaner` $\rightarrow$ `TF-IDF` $\rightarrow$ `LinearSVC` $\rightarrow$ `LLM Gemini`.
  * **Khối API & Web UI:** Dựng FastAPI Backend & Streamlit Web App trực quan.
  * **Đóng gói & Quản trị:** Docker Compose, hoàn thiện Slide và Báo cáo BTL trong `docs/report/`.

---

## 🚀 V. CẨM NANG GIT & GITHUB TOÀN TẬP (DÀNH CHO NGƯỜI CHƯA BIẾT GÌ)

> **Dành cho các thành viên mới:** Đừng lo lắng! Hãy đọc kỹ từng bước dưới đây và làm theo thứ tự. Bạn chỉ cần thao tác trên nhánh riêng của mình, không sợ làm hỏng code của nhóm.

---

### 1. Hiểu bản chất Git trong 30 giây (Mô hình 3 trạm)

Khi bạn làm việc trên máy tính của mình:
```
[File trên máy bạn]  ──(git add .)──>  [Hòm thư chờ gửi]  ──(git commit)──>  [Két sắt trên máy bạn]  ──(git push)──>  [Kho chung GitHub]
(Working Directory)                     (Staging Area)                        (Local Repository)                      (Remote GitHub)
```
* `git add .` : Nhặt tất cả các file bạn vừa sửa/tạo mới bỏ vào "Hòm thư chờ gửi".
* `git commit -m "tin_nhan"` : Niêm phong hòm thư, ghi nhãn công việc bạn vừa làm.
* `git push` : Chuyển hòm thư đó lên trang web GitHub của nhóm.

---

### 2. Giai đoạn 1: Cài đặt và Tải dự án về máy (Chỉ làm duy nhất 1 lần đầu)

#### Bước 1.1: Kiểm tra máy đã cài Git chưa
Mở **Command Prompt (cmd)** hoặc **PowerShell** trên Windows, gõ:
```bash
git --version
```
* Nếu hiện ra phiên bản (ví dụ `git version 2.40...`) $\rightarrow$ Đã có Git.
* Nếu báo lỗi không nhận lệnh $\rightarrow$ Truy cập https://git-scm.com tải bản Windows về bấm Next cài đặt.

#### Bước 1.2: Khai báo tên và email của bạn (Bắt buộc để GitHub ghi nhận bạn)
Chạy lần lượt 2 lệnh sau (thay tên và email thật của bạn):
```bash
git config --global user.name "Nguyen Van A"
git config --global user.email "nguyenvana@gmail.com"
```

#### Bước 1.3: Tải mã nguồn dự án về máy tính
1. Tạo một thư mục trên máy bạn (ví dụ ổ `D:\HocTap`).
2. Mở thư mục đó bằng **VS Code** (hoặc mở Terminal tại đó).
3. Gõ lệnh clone:
```bash
git clone <LINK_REPO_GITHUB_DO_NAM_GUI>
cd IT3190_Project
```

#### Bước 1.4: Chuyển sang ĐÚNG nhánh được phân công của bạn
**RẤT QUAN TRỌNG:** Tuyệt đối không ở nhánh `main`. Hãy chuyển sang nhánh của bạn:
```bash
# Nếu bạn là Mai Hân:
git checkout feature/data-preprocessing

# Nếu bạn là Đình Khôi:
git checkout feature/ml-linear-models

# Nếu bạn là Trung Kiên:
git checkout feature/ml-tree-models

# Nếu bạn là Hoàng Anh:
git checkout feature/model-evaluation
```
Kiểm tra lại xem mình đang ở đúng nhánh chưa:
```bash
git branch
```
👉 Bạn thấy dấu sao màu xanh `*` nằm ở đúng tên nhánh của bạn là thành công!

---

### 3. Giai đoạn 2: Vòng lặp làm việc hàng ngày (Daily Workflow)

Mỗi ngày ngồi vào bàn học để code, bạn thực hiện đúng 5 bước sau:

#### Bước 2.1: Lấy code mới nhất từ Nhóm trưởng (Tránh bị lạc hậu code)
Trước khi gõ bất kỳ dòng code nào, hãy cập nhật code mới từ nhánh `main`:
```bash
git pull origin main
```

#### Bước 2.2: Tiến hành viết code
Mở file trong thư mục được phân công của bạn trên VS Code và viết code/chạy notebook bình thường.

#### Bước 2.3: Xem danh sách các file bạn vừa chỉnh sửa
Khi làm xong, mở Terminal gõ:
```bash
git status
```
* Git sẽ hiện danh sách các file màu đỏ (nghĩa là file đã bị sửa hoặc mới tạo).

#### Bước 2.4: Đóng gói các thay đổi (Add & Commit)
Chạy lần lượt 2 lệnh:
```bash
git add .
git commit -m "feat: mo ta ngan gon viec ban vua lam"
```
*Ví dụ:* `git commit -m "feat: hoan thanh xong ham clean_text trong text_cleaner"`

#### Bước 2.5: Đẩy code lên GitHub
```bash
git push origin <ten-nhanh-cua-ban>
```
*Ví dụ Mai Hân:* `git push origin feature/data-preprocessing`  
*Ví dụ Đình Khôi:* `git push origin feature/ml-linear-models`

---

### 4. Giai đoạn 3: Tạo Pull Request (PR) trên web GitHub để Nhóm trưởng duyệt

Sau khi `git push` xong, code mới chỉ nằm ở nhánh của bạn, chưa vào nhánh chính. Bạn cần tạo Pull Request:

1. Mở trình duyệt web, vào link repo GitHub của nhóm.
2. Bạn sẽ thấy một thanh thông báo màu vàng nổi lên ngay đầu trang có chữ: **"feature/... had recent pushes"**.
3. Bấm vào nút màu xanh lá cây: **Compare & pull request**.
4. Kiểm tra 2 ô chọn nhánh ở đầu trang:
   * `base: main` $\leftarrow$ `compare: <nhanh-cua-ban>` (phải mũi tên chỉ về main).
5. Ô tiêu đề ghi ngắn gọn những gì bạn đã làm.
6. Bấm nút xanh lá cây lớn: **Create pull request**.
7. Nhắn tin vào nhóm Zalo: *"Nam ơi tớ làm xong task và tạo PR rồi, bạn vào review giúp tớ nhé!"*. Nhóm trưởng sẽ kiểm tra code và bấm nút merge vào `main`!

---

### 5. Giai đoạn 4: "Cẩm nang cấp cứu" khi gặp sự cố Git

| Tình huống / Sự cố | Nguyên nhân | Cách xử lý an toàn nhất |
| :--- | :--- | :--- |
| **Lỡ sửa nhầm file không phải của mình** | Mở nhầm file của người khác sửa linh tinh | Gõ lệnh hủy thay đổi file đó: `git restore <duong_dan_file>` |
| **Gõ git push bị báo lỗi `[rejected]`** | Trên GitHub có code mới hơn mà máy bạn chưa tải về | Gõ `git pull origin <nhanh_cua_ban>` rồi gõ lại lệnh push |
| **Không biết mình đang đứng ở nhánh nào** | Hay quên nhánh | Gõ `git branch` để xem nhánh hiện tại (có dấu `*`) |
| **Báo lỗi chữ đỏ CONFLICT hoặc màn hình lạ** | Bị xung đột nội dung file | **DỪNG LẠI NGAY!** Không gõ bừa lệnh xóa. Chụp toàn bộ màn hình gửi ngay cho Nhóm trưởng Hải Nam để được hỗ trợ gỡ rối! |
