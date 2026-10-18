# 📜 CLEAN ARCHITECTURE & AGENT-CODING RULES

## 1. Ranh giới Kiến trúc (Clean Architecture Boundary)
- **Tầng Domain (`source/domain/`):** Tuyệt đối KHÔNG import bất kỳ thư viện framework ngoài nào (cấm `sklearn`, `fastapi`, `streamlit`, `pandas`, `google`). Chỉ sử dụng thư viện chuẩn của Python (`dataclasses`, `enum`, `typing`, `abc`) và `numpy` cho ma trận.
- **Tầng Services (`source/services/cli/`):** Là Bộ điều phối chính duy nhất (Primary Orchestrator). Chỉ phụ thuộc vào Domain Interface và Entities. Không được import trực tiếp FastAPI hoặc Streamlit.
- **Tầng Infrastructure/Adapters (`source/infrastructure/`, `source/model/`, `source/data_process/`):** Nơi cài đặt cụ thể các interface của Domain. Được phép dùng `sklearn`, `underthesea`, `joblib`.
- **Tầng Presentation (`source/api/`, `source/ui/`):** Chỉ gọi Use Cases từ tầng Services và nhận về Domain Entities/DTOs. Cấm gọi trực tiếp Model và Preprocessing.

## 2. Rào chắn Cú pháp & Thói quen Viết code
- **Cấm hàm `print()`:** Tuyệt đối dùng `get_channel_logger(LogChannel)` theo kênh thích hợp. Chỉ file giao diện dòng lệnh `source/services/cli/main.py` được dùng `print` để hiển thị cho người dùng.
- **Cấm comment rác:** Không viết `# TODO`, `# Step 1: ...`, comment giải thích vụn vặt. Sử dụng docstring chuẩn Google/Sphinx.
- **Bắt buộc Type Annotations:** Mọi hàm, phương thức đều phải có đầy đủ kiểu dữ liệu tham số và giá trị trả về.
- **Không hardcode đường dẫn:** Luôn sử dụng `paths` từ `source/infrastructure/config.py`.

## 3. Quy trình Kiểm chứng Khép kín (Closed-Loop)
- Trước khi báo cáo hoàn thành nhiệm vụ, Agent BẮT BUỘC phải chạy:
  ```bash
  python scripts/verify_loop.py
  ```
- Nếu có giao diện Streamlit/API, Agent BẮT BUỘC sử dụng **Chrome DevTools MCP** hoặc `TestClient` để khởi chạy, kiểm tra console log và chụp ảnh màn hình làm bằng chứng.

## 4. Nguyên tắc Luồng Dữ liệu Duy nhất (Single Data Flow Rule)
- Dữ liệu thô (`data/raw_data/`) và dữ liệu sạch (`data/clean_data/`) **CHỈ ĐƯỢC PHÉP TRUY CẬP** qua `source/infrastructure/database/repo/` (triển khai từ `BaseDatasetRepository` tại Domain).
- Tầng `source/services/cli/` là tầng duy nhất gọi Repository để đọc/ghi dữ liệu, sau đó truyền trực tiếp `(X, y)` vào các Model và Preprocessor.
- **CẤM TUYỆT ĐỐI**: Các module trong `source/model/` và `source/data_process/` không được tự ý mở file JSON từ đĩa hoặc import trực tiếp `infrastructure.database`.

## 5. Nguyên tắc Trực quan hóa Miền (Domain Visualization Rule)
- Toàn bộ thuật toán sinh biểu đồ phân tích EDA, tuning siêu tham số và đánh giá đối đầu Best Models sống độc lập tại `source/domain/visualization/`.
- Biểu đồ chỉ nhận dữ liệu thuần (mảng numpy, list, dict, entities), xuất ảnh vào `docs/figures/` theo 3 thư mục phân cấp (`data_preprocessing/`, `hyperparameter_tuning/`, `model_comparison/`).
- Các hàm vẽ biểu đồ chỉ được điều phối và kích hoạt thông qua lệnh dòng lệnh CLI tại `source/services/cli/main.py`.

## 6. Hệ thống Phân kênh Logging Tách biệt (Separated Structured Logging Rule)
- Tuyệt đối không ghi dồn tất cả log vào một file hỗn độn. Mọi nghiệp vụ phải ghi đúng kênh qua `get_channel_logger(LogChannel)`:
  - `LogChannel.DATA_PIPELINE` -> `logs/data_pipeline.log`
  - `LogChannel.MODEL_TRAINING` -> `logs/model_training.log`
  - `LogChannel.APP_SERVICE` -> `logs/app_service.log`
  - `LogChannel.TEST_EXECUTION` -> `logs/test_execution.log`
- Mỗi file log sử dụng `RotatingFileHandler` (tối đa 5MB, lưu 3 file backup).
