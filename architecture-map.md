# 🗺️ ARCHITECTURE & CAPABILITY MAP (IT3190_Project)

> **Dành cho AI Agent & Thành viên:** Bản đồ định vị giúp tra cứu tức thì khi nhận lỗi, tạo tính năng mới hoặc kiểm chứng tự động. Mọi thay đổi code PHẢI tuân thủ hợp đồng tại tầng Domain tương ứng và đi qua Bộ điều phối chính `services/cli`.

---

## 📌 BẢN ĐỒ ÁNH XẠ KHỐI CHỨC NĂNG (TINH GỌN & CHUẨN MỰC)

```
┌────────────────────────────────────────────────────────┐
│ TẦNG 4: PRESENTATION (source/api/, source/ui/)         │
│ • FastAPI REST API: source/api/main.py                 │
│ • Streamlit Web UI: source/ui/app.py                   │
│ ❌ CẤM GỌI TRỰC TIẾP VÀO MODEL HOẶC PREPROCESSING     │
└───────────────────────┬────────────────────────────────┘
                        │ (BẮT BUỘC GỌI DUY NHẤT VÀO ORCHESTRATOR)
                        ▼
┌────────────────────────────────────────────────────────┐
│ TẦNG 2: BỘ ĐIỀU PHỐI CHÍNH & CLI (source/services/cli/)│
│ • Orchestrator: SentimentPipelineOrchestrator          │
│ • Training Pipeline: PipelineTrainingService           │
│ • CLI Entrypoint: source/services/cli/main.py          │
│ • Tương tác trực tiếp với Domain Layer qua Interface   │
└───────────┬───────────────────────────────┬────────────┘
            │                               │
            ▼                               ▼
┌───────────────────────────┐ ┌──────────────────────────────────────────┐
│ TẦNG 1: DOMAIN CORE       │ │ TẦNG 3: CONCRETE IMPLEMENTATIONS & ADAPT │
│ (source/domain/)          │ │ • Infrastructure/Database/Repo:          │
│ • BasePreprocess          │ │   JSONDatasetRepository                  │
│ • BaseSentimentClassifier │ │ • Infrastructure/Logging:                │
│ • BaseDatasetRepository   │ │   Multi-channel Separated Logging        │
│ • EmotionScoreProfile     │ │ • Preprocessing: TextCleaner             │
│ • Visualization:          │ │ • Featuring: TfidfFeatureExtractor       │
│   EDA, Tuning, Benchmark  │ │ • Models: Linear, Tree Algorithms        │
└───────────────────────────┘ └──────────────────────────────────────────┘
```

---

## 🔒 NGUYÊN TẮC LUỒNG DỮ LIỆU & LOGGING PHÂN TẦNG

1. **Nguyên tắc Dòng chảy Dữ liệu (Strict Data Flow)**:
   ```
   data/raw_data/*.json / data/clean_data/*.json
         ▲
         │ (Đọc/Ghi duy nhất qua Repository)
         ▼
   JSONDatasetRepository (source/infrastructure/database/repo/)
         ▲
         │ (Trả dữ liệu về Service)
         ▼
   PipelineTrainingService (source/services/cli/training_service.py)
         │
         ├─── Raw Text ───► TextCleaner (data_process/preprocessing/)
         │
         ├─── Clean Text ─► TfidfFeatureExtractor (data_process/featuring/)
         │
         └─── Matrix (X, y) ─► Classifier Models (source/model/classifier/)
   ```
   - **Rào chắn cứng (`.importlinter`)**: `source.model` và `source.data_process` CẤM TUYỆT ĐỐI không được import `source.infrastructure.database`.

2. **Hệ thống Logging Đa Kênh Tách Biệt (`logs/`)**:
   - `logs/data_pipeline.log`: Log quá trình làm sạch teencode/emoji, lưu clean data, trích xuất TF-IDF.
   - `logs/model_training.log`: Log quá trình GridSearch siêu tham số, loss, macro-f1, thời gian train.
   - `logs/app_service.log`: Log hoạt động của Orchestrator, CLI runner, API server, Streamlit web.
   - `logs/test_execution.log`: Log chi tiết quá trình kiểm thử tự động của Pytest.

---

## 📍 CHI TIẾT TỪNG KHỐI CHỨC NĂNG

### 1. Khối Tiền xử lý dữ liệu (Preprocessing Subsystem)
- **Domain Contract:** `source/domain/interface/base_preprocess.py` (`BasePreprocess`)
- **Domain Entity:** `source/domain/entities/sentiment_sample.py` (`CleanedTextSample`)
- **Thực thi:** `source/data_process/preprocessing/text_cleaner.py` (`TextCleaner`), `teencode_dict.py`, `emoji_dict.py`
- **Tập dữ liệu:** Chỉ truy cập gián tiếp qua `JSONDatasetRepository`.
- **File Test:** `tests/unit/test_cleaner.py`

### 2. Khối Trích xuất Đặc trưng (Featuring Subsystem)
- **Thực thi:** `source/data_process/featuring/tfidf_extractor.py` (`TfidfFeatureExtractor`), `label_encoder.py`
- **Cấu hình:** TF-IDF N-grams (1, 2), `max_features=5000`, `min_df=2`, `sublinear_tf=True`.
- **Nguyên tắc:** Fit DUY NHẤT trên Train set, transform trên Valid và Test.
- **Artifact lưu trữ:** `source/model/classifier/saved_models/tfidf_vectorizer.joblib`
- **File Test:** `tests/unit/test_featuring.py`

### 3. Khối Mô hình Học máy & Ước lượng Thang điểm 7 Chiều (Classifier Subsystem)
- **Domain Contract:** `source/domain/interface/base_classifer.py` (`BaseSentimentClassifier`)
- **Domain Entity:** `source/domain/entities/sentiment_sample.py` (`EmotionScoreProfile`, `EmotionLabel`)
- **5 Mô hình thực nghiệm:**
  - `MultinomialNBClassifier`: Laplace smoothing $\alpha \in [0.01, 2.0]$.
  - `LogisticRegressionClassifier`: $C \in [0.01, 10.0]$, `class_weight='balanced'`.
  - `LinearSVCClassifier`: Calibrated with Platt Scaling (`CalibratedClassifierCV(cv=3)`), $C \in [0.01, 10.0]$.
  - `DecisionTreeModel`: `max_depth` $\in [10, 30, \text{None}]$, `min_samples_split` $\in [2, 10]$.
  - `RandomForestModel`: `n_estimators` $\in [50, 200]$, `max_depth` $\in [15, 35, \text{None}]$.
- **Artifact lưu trữ:** `source/model/classifier/saved_models/best_model.joblib`
- **File Test:** `tests/unit/test_classifiers.py`

### 4. Khối Trực quan hóa Miền (Domain Visualization Subsystem)
- **Thực thi:** `source/domain/visualization/`
  - `eda_visualizer.py`: Vẽ so sánh dữ liệu thô vs sạch $\rightarrow$ lưu `docs/figures/data_preprocessing/` (3 biểu đồ).
  - `tuning_visualizer.py`: Vẽ đường cong tuning tham số từng mô hình $\rightarrow$ lưu `docs/figures/hyperparameter_tuning/` (6 biểu đồ).
  - `benchmark_visualizer.py`: Vẽ đối đầu Best Models, Confusion Matrix, ROC curves, Radar chart $\rightarrow$ lưu `docs/figures/model_comparison/` (5 biểu đồ).
- **File Test:** `tests/unit/test_domain_visualization.py`

### 5. Khối Điều phối Chính & Dòng lệnh (Primary Orchestrator & CLI Runner)
- **Thực thi:**
  - Class điều phối suy luận: `source/services/cli/orchestrator.py` (`SentimentPipelineOrchestrator`)
  - Class điều phối huấn luyện: `source/services/cli/training_service.py` (`PipelineTrainingService`)
  - CLI Runner: `source/services/cli/main.py`
- **Các lệnh CLI chính:**
  ```bash
  # 1. Làm sạch dữ liệu
  python -m source.services.cli.main clean-data

  # 2. Sinh biểu đồ phân tích so sánh dữ liệu
  python -m source.services.cli.main eda-charts

  # 3. Thử nghiệm tối ưu siêu tham số cho 5 mô hình
  python -m source.services.cli.main tune-models

  # 4. Đánh giá đối đầu các Best Models và lưu best_model.joblib
  python -m source.services.cli.main evaluate-best

  # 5. Chạy toàn bộ pipeline từ A-Z
  python -m source.services.cli.main run-all

  # 6. Dự đoán cảm xúc (với model thật đã huấn luyện)
  python -m source.services.cli.main predict --text "Đồ ăn rất ngon phục vụ nhiệt tình 10 điểm ❤️"

  # 7. Kiểm tra sức khỏe toàn bộ Clean Architecture
  python -m source.services.cli.main health
  ```
- **File Test:** `tests/unit/test_cli_orchestrator.py`

### 6. Khối Cấu hình Tập trung Toàn cục (Centralized Configuration)
- **Thực thi:** `source/config.py` (tái xuất tại `config.py` gốc và `source/infrastructure/config.py` tương thích ngược)
  - `ProjectPaths`: Quản lý tập trung mọi đường dẫn thư mục chuẩn (`root`, `source`, `data_raw`, `data_clean`, `models_saved`, `docs`, `figures`, `logs`).
  - `Settings`: Đọc tham số môi trường từ `.env` (`app_name`, `api_port`, `streamlit_port`, `log_level`).

### 7. Đặc tả Đóng gói Container & Triển khai Render Cloud
- **Cấu hình Docker:**
  - `Dockerfile`: Multi-stage Python 3.11-slim, chạy dưới user `appuser` không đặc quyền root, tích hợp Healthcheck.
  - `.dockerignore`: Loại bỏ triệt để context thừa (`.git`, caches, docs, tests) giúp context build siêu nhẹ (< 50MB).
  - `deploy/docker-compose.yml`: Điều phối 2 container cục bộ `api` (port 8000) và `ui` (port 8501).
  - `deploy/render.yaml`: Blueprint Infrastructure-as-Code triển khai tự động 1-click lên Render Web Service.
- **Ước lượng Tài nguyên & Sizing cho Render:**
  - **Dung lượng RAM thực tế:** Uvicorn + Pipeline TF-IDF + LogisticRegression tiêu thụ **~195 MB RAM**.
  - **Tốc độ phản hồi:** Độ trễ trung bình **5.03 ms**, QPS đạt **~195 requests/s**.
  - **Render Free Web Service Tier (512 MB RAM, 0.1 CPU):** Hoàn toàn đáp ứng xuất sắc (sử dụng ~38% RAM, dư dả > 300 MB).

