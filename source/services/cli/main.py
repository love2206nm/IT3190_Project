"""Command Line Interface Entrypoint for IT3190 Project.

Cung cấp công cụ dòng lệnh điều phối toàn bộ đường ống phân loại cảm xúc:
- predict: Dự đoán cảm xúc từ chuỗi văn bản (sử dụng Best Model nếu có).
- clean-data: Tiền xử lý dữ liệu và lưu vào data/clean_data/.
- eda-charts: Sinh 3 biểu đồ so sánh dữ liệu thô vs sạch lưu vào docs/figures/data_preprocessing/.
- tune-models: Thử nghiệm tối ưu siêu tham số cho 5 mô hình lưu vào docs/figures/hyperparameter_tuning/.
- evaluate-best: Đánh giá đối đầu 5 Best Models trên tập Test, lưu vào docs/figures/model_comparison/
  và lưu best_model.joblib.
- run-all: Tự động chạy toàn bộ quy trình từ A-Z một lệnh duy nhất.
- health: Kiểm tra tình trạng sức khỏe các tầng Clean Architecture.
"""

import argparse
import sys

from source.config import paths, settings
from source.services.cli.orchestrator import SentimentPipelineOrchestrator
from source.services.cli.training_service import PipelineTrainingService

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def render_ascii_bar(score: float, width: int = 25) -> str:
    """Tạo thanh biểu đồ ASCII trực quan hóa cường độ."""
    filled = round(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score * 100:.1f}%"


def handle_predict(args: argparse.Namespace) -> int:
    """Xử lý lệnh dự đoán cảm xúc cho một câu văn bản."""
    text = args.text
    if not text or not text.strip():
        print("[LỖI] Nội dung văn bản không được để trống.")
        return 1

    orchestrator = SentimentPipelineOrchestrator.build_default()
    profile = orchestrator.analyze_text(text)

    print("\n" + "=" * 65)
    print("🎭 KẾT QUẢ THANG ĐO PHỔ ĐIỂM 7 CHIỀU CẢM XÚC (UIT-VSMEC)")
    print("=" * 65)
    print(f"📝 Văn bản: {profile.text}")
    print(f"🎯 Cảm xúc chủ đạo: {profile.dominant_emotion.value} (Cường độ: {profile.intensity * 100:.1f}%)")

    if profile.is_mixed:
        print("⚠️ CẢNH BÁO: Phát hiện cảm xúc pha trộn (Mixed Emotions)!")

    print("\n📊 Phân bố cường độ 7 sắc thái cảm xúc:")
    print("-" * 65)
    for emotion, score in profile.ranking:
        bar_str = render_ascii_bar(score)
        print(f"  • {emotion.value:<12} {bar_str}")
    print("=" * 65 + "\n")
    return 0


def handle_clean_data(args: argparse.Namespace) -> int:
    """Xử lý lệnh tiền xử lý dữ liệu từ data/raw_data/ -> data/clean_data/."""
    print("\n[BƯỚC 1] Khởi chạy tiền xử lý dữ liệu văn bản mạng xã hội (UIT-VSMEC)...")
    service = PipelineTrainingService()
    counts = service.clean_and_save_data()
    print("✅ Hoàn tất làm sạch và chuẩn hóa dữ liệu:")
    for split, count in counts.items():
        print(f"  • {split:<6}: {count} mẫu sạch đã lưu vào data/clean_data/{split}.json")
    print()
    return 0


def handle_eda_charts(args: argparse.Namespace) -> int:
    """Xử lý lệnh sinh 3 biểu đồ EDA so sánh dữ liệu thô vs sạch."""
    print("\n[BƯỚC 2] Sinh các biểu đồ EDA so sánh dữ liệu vào docs/figures/data_preprocessing/...")
    service = PipelineTrainingService()
    figs = service.generate_eda_charts()
    print("✅ Đã xuất bản thành công 3 biểu đồ phân tích dữ liệu:")
    for f in figs:
        print(f"  • {f.name}")
    print()
    return 0


def handle_tune_models(args: argparse.Namespace) -> int:
    """Xử lý lệnh thử nghiệm tối ưu siêu tham số cho 5 mô hình."""
    print("\n[BƯỚC 3] Chạy thử nghiệm lưới siêu tham số (Hyperparameter Tuning)...")
    service = PipelineTrainingService()
    results = service.run_hyperparameter_tuning()
    print("✅ Hoàn tất thử nghiệm và xuất đồ thị tuning vào docs/figures/hyperparameter_tuning/:")
    for model_name, info in results.items():
        param_str = ", ".join(f"{k}={v}" for k, v in info["best_param"].items())
        f1 = info["val_metrics"]["macro_f1"]
        print(f"  • {model_name:<20}: Tham số tối ưu [{param_str}] | Val Macro-F1: {f1:.4f}")
    print()
    return 0


def handle_evaluate_best(args: argparse.Namespace) -> int:
    """Xử lý lệnh đánh giá đối đầu 5 Best Models và lưu best_model.joblib."""
    print("\n[BƯỚC 4] Đánh giá đối đầu 5 Best Models trên tập Test và lưu trữ mô hình tốt nhất...")
    service = PipelineTrainingService()
    report = service.evaluate_and_save_best_models()
    print("\n" + "=" * 65)
    print(f"🏆 MÔ HÌNH VÔ ĐỊCH: {report['best_model_name']} (Test Macro-F1: {report['best_macro_f1']:.4f})")
    print("=" * 65)
    for m, met in report["models"].items():
        f1_str = f"Macro-F1: {met['macro_f1']:.4f} | Acc: {met['accuracy']:.4f}"
        loss_str = f"LogLoss: {met['log_loss']:.4f} | Time: {met['train_time_sec']:.2f}s"
        print(f"  • {m:<25}: {f1_str} | {loss_str}")
    print("=" * 65)
    print("✅ Đã xuất 5 biểu đồ đối đầu vào: docs/figures/model_comparison/")
    print("✅ Đã lưu trọng số vào: source/model/classifier/saved_models/best_model.joblib\n")
    return 0


def handle_run_all(args: argparse.Namespace) -> int:
    """Xử lý lệnh chạy toàn bộ pipeline tự động từ đầu đến cuối."""
    print("\n" + "=" * 65)
    print("🚀 BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE CLEAN ARCHITECTURE (A -> Z)")
    print("=" * 65)
    service = PipelineTrainingService()
    res = service.run_full_pipeline()
    print("\n" + "=" * 65)
    print("🎉 HOÀN TẤT 100% PIPELINE:")
    print(f"  • Tổng số mẫu làm sạch: {res['clean_counts']}")
    print("  • 3 Biểu đồ EDA: docs/figures/data_preprocessing/")
    print("  • 6 Biểu đồ Tuning: docs/figures/hyperparameter_tuning/")
    print("  • 5 Biểu đồ Đối đầu: docs/figures/model_comparison/")
    print(f"  • Mô hình tốt nhất: {res['best_model']} (Test Macro-F1: {res['best_macro_f1']:.4f})")
    print("=" * 65 + "\n")
    return 0


def handle_health(args: argparse.Namespace) -> int:
    """Kiểm tra tình trạng sức khỏe các tầng trong Clean Architecture."""
    print("\n" + "=" * 60)
    print("🏥 KIỂM TRA TRẠNG THÁI HỆ THỐNG CLEAN ARCHITECTURE")
    print("=" * 60)
    print(f"  • Ứng dụng: {settings.app_name} (v{settings.app_version})")
    print(f"  • Thư mục gốc: {paths.root}")
    print(f"  • Thư mục Dữ liệu sạch: {paths.data_clean} (Tồn tại: {paths.data_clean.exists()})")
    print(f"  • Thư mục Models lưu trữ: {paths.models_saved} (Tồn tại: {paths.models_saved.exists()})")
    print(f"  • Thư mục Báo cáo/Đồ thị: {paths.figures} (Tồn tại: {paths.figures.exists()})")

    # Kiểm tra khởi tạo orchestrator
    orchestrator = SentimentPipelineOrchestrator.build_default()
    test_profile = orchestrator.analyze_text("Kiểm tra kết nối hệ thống")
    print(f"  • Bộ điều phối chính (Orchestrator): HOẠT ĐỘNG TỐT (Test score: {len(test_profile.scores)} nhãn)")
    print("=" * 60)
    print("✅ HỆ THỐNG SẴN SÀNG PHỤC VỤ DÒNG LỆNH, API VÀ STREAMLIT.\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m source.services.cli.main",
        description="Bộ điều khiển dòng lệnh trung tâm của hệ thống phân loại cảm xúc UIT-VSMEC.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực thi")

    # 1. predict
    predict_parser = subparsers.add_parser("predict", help="Dự đoán phổ điểm 7 chiều cho câu văn bản")
    predict_parser.add_argument("--text", type=str, required=True, help="Câu tiếng Việt cần phân tích cảm xúc")

    # 2. clean-data
    subparsers.add_parser("clean-data", help="Tiền xử lý dữ liệu và lưu vào data/clean_data/")

    # 3. eda-charts
    subparsers.add_parser("eda-charts", help="Sinh 3 biểu đồ so sánh dữ liệu thô vs sạch")

    # 4. tune-models
    subparsers.add_parser("tune-models", help="Thử nghiệm tối ưu siêu tham số cho 5 mô hình")

    # 5. evaluate-best
    subparsers.add_parser("evaluate-best", help="Đánh giá đối đầu 5 Best Models và lưu best_model.joblib")

    # 6. run-all
    subparsers.add_parser("run-all", help="Chạy toàn bộ pipeline từ A-Z một lệnh duy nhất")

    # 7. health
    subparsers.add_parser("health", help="Kiểm tra trạng thái các khối chức năng của hệ thống")

    args = parser.parse_args()

    if args.command == "predict":
        return handle_predict(args)
    if args.command == "clean-data":
        return handle_clean_data(args)
    if args.command == "eda-charts":
        return handle_eda_charts(args)
    if args.command == "tune-models":
        return handle_tune_models(args)
    if args.command == "evaluate-best":
        return handle_evaluate_best(args)
    if args.command == "run-all":
        return handle_run_all(args)
    if args.command == "health":
        return handle_health(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
