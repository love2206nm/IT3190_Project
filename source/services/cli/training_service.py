"""Pipeline Training & Experimentation Service (Services Layer).

Đóng vai trò điều phối toàn bộ quy trình từ dữ liệu -> tiền xử lý -> trích xuất đặc trưng ->
thử nghiệm tối ưu siêu tham số -> đánh giá đối đầu -> xuất bản các biểu đồ trực quan hóa.
Tuân thủ Clean Architecture: Dữ liệu duy nhất đi từ Database Repository -> Service -> Model.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from source.config import paths
from source.data_process.featuring.label_encoder import ORDERED_EMOTIONS, LabelEncoderHelper
from source.data_process.featuring.tfidf_extractor import TfidfFeatureExtractor
from source.data_process.preprocessing.text_cleaner import TextCleaner
from source.domain.interface.base_dataset_repo import BaseDatasetRepository
from source.domain.interface.base_preprocess import BasePreprocess
from source.domain.visualization.benchmark_visualizer import BenchmarkVisualizer
from source.domain.visualization.eda_visualizer import EDAVisualizer
from source.domain.visualization.tuning_visualizer import TuningVisualizer
from source.infrastructure.database.repo.json_dataset_repo import JSONDatasetRepository
from source.infrastructure.logging import LogChannel, get_channel_logger
from source.model.classifier.algorithms.linear_models import (
    LinearSVCClassifier,
    LogisticRegressionClassifier,
    MultinomialNBClassifier,
)
from source.model.classifier.algorithms.tree_models import (
    DecisionTreeModel,
    RandomForestModel,
)
from source.model.classifier.evaluator import ModelEvaluator

logger_pipe = get_channel_logger(LogChannel.DATA_PIPELINE, name="training_service")
logger_model = get_channel_logger(LogChannel.MODEL_TRAINING, name="training_service")


class PipelineTrainingService:
    """Dịch vụ điều phối huấn luyện mô hình và tạo báo cáo trực quan hóa."""

    def __init__(
        self,
        repo: Optional[BaseDatasetRepository] = None,
        cleaner: Optional[BasePreprocess] = None,
        vectorizer: Optional[TfidfFeatureExtractor] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        """Khởi tạo dịch vụ với Dependency Injection."""
        if project_root is None:
            self.root_dir = paths.root
        else:
            self.root_dir = project_root

        self.repo = repo if repo else JSONDatasetRepository(self.root_dir)
        self.cleaner = cleaner if cleaner else TextCleaner()
        self.vectorizer = vectorizer if vectorizer else TfidfFeatureExtractor(max_features=5000)

        # Các đường dẫn output
        self.figures_dir = self.root_dir / "docs" / "figures"
        self.saved_models_dir = self.root_dir / "source" / "model" / "classifier" / "saved_models"
        self.docs_dir = self.root_dir / "docs"

        self.saved_models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. TIỀN XỬ LÝ VÀ LƯU DỮ LIỆU SẠCH (RAW -> CLEAN)
    # =========================================================================
    def clean_and_save_data(self) -> Dict[str, int]:
        """Đọc raw data qua repository -> làm sạch qua TextCleaner -> lưu clean data."""
        logger_pipe.info("=== BẮT ĐẦU QUY TRÌNH LÀM SẠCH DỮ LIỆU TOÀN DIỆN ===")
        counts: Dict[str, int] = {}

        for split in ["train", "valid", "test"]:
            raw_samples = self.repo.load_raw_dataset(split)
            logger_pipe.info(f"Đang làm sạch tập '{split}' ({len(raw_samples)} mẫu)...")
            cleaned_samples = self.cleaner.clean_dataset(raw_samples)
            self.repo.save_clean_dataset(split, cleaned_samples)
            counts[split] = len(cleaned_samples)

        logger_pipe.info(f"Hoàn tất làm sạch cả 3 tập: {counts}")
        return counts

    # =========================================================================
    # 2. XUẤT BỘ 3 BIỂU ĐỒ EDA SO SÁNH DỮ LIỆU (docs/figures/data_preprocessing/)
    # =========================================================================
    def generate_eda_charts(self) -> List[Path]:
        """Tạo 3 biểu đồ phân tích EDA dữ liệu thô vs sạch."""
        logger_pipe.info("=== BẮT ĐẦU SINH BỘ BIỂU ĐỒ EDA SO SÁNH DỮ LIỆU ===")
        eda_dir = self.figures_dir / "data_preprocessing"
        eda_dir.mkdir(parents=True, exist_ok=True)

        # Lấy dữ liệu thô và sạch qua repository
        train_raw = self.repo.load_raw_dataset("train")
        valid_raw = self.repo.load_raw_dataset("valid")
        test_raw = self.repo.load_raw_dataset("test")

        train_clean = self.repo.load_clean_dataset("train")
        valid_clean = self.repo.load_clean_dataset("valid")
        test_clean = self.repo.load_clean_dataset("test")

        raw_texts = [s.get("Sentence", "") for s in (train_raw + valid_raw + test_raw)]
        clean_texts = [s.get("Sentence", "") for s in (train_clean + valid_clean + test_clean)]

        # 1. Biểu đồ phân bố độ dài câu
        p1 = eda_dir / "01_sentence_length_distribution.png"
        EDAVisualizer.plot_sentence_length_distribution(raw_texts, clean_texts, p1)
        logger_pipe.info(f"Đã xuất: {p1}")

        # 2. Biểu đồ so sánh từ vựng & token emoji
        raw_words = set(" ".join(raw_texts).split())
        clean_words = set(" ".join(clean_texts).split())

        # Thống kê tần suất semantic tokens (icon_*)
        all_clean_words = " ".join(clean_texts).split()
        emoji_tokens = [w for w in all_clean_words if w.startswith("icon_")]
        emoji_counts: Dict[str, int] = {}
        for tok in emoji_tokens:
            emoji_counts[tok] = emoji_counts.get(tok, 0) + 1

        p2 = eda_dir / "02_vocabulary_and_sentiment_tokens.png"
        EDAVisualizer.plot_vocabulary_and_sentiment_tokens(
            raw_vocab_size=len(raw_words),
            clean_vocab_size=len(clean_words),
            emoji_counts=emoji_counts,
            output_path=p2,
        )
        logger_pipe.info(f"Đã xuất: {p2}")

        # 3. Biểu đồ phân bố 7 nhãn trên Train/Valid/Test
        train_labels = [s.get("Emotion", "Other") for s in train_clean]
        valid_labels = [s.get("Emotion", "Other") for s in valid_clean]
        test_labels = [s.get("Emotion", "Other") for s in test_clean]

        p3 = eda_dir / "03_dataset_split_and_label_distribution.png"
        EDAVisualizer.plot_dataset_label_distribution(train_labels, valid_labels, test_labels, p3)
        logger_pipe.info(f"Đã xuất: {p3}")

        return [p1, p2, p3]

    # =========================================================================
    # 3. THỬ NGHIỆM SIÊU THAM SỐ (docs/figures/hyperparameter_tuning/)
    # =========================================================================
    def run_hyperparameter_tuning(self) -> Dict[str, Any]:
        """Thử nghiệm lưới tham số cho 5 mô hình và vẽ các biểu đồ tuning."""
        logger_model.info("=== BẮT ĐẦU QUY TRÌNH THỬ NGHIỆM TỐI ƯU SIÊU THAM SỐ ===")
        tune_dir = self.figures_dir / "hyperparameter_tuning"
        tune_dir.mkdir(parents=True, exist_ok=True)

        # 1. Đọc dữ liệu sạch và chuẩn bị ma trận TF-IDF
        train_clean = self.repo.load_clean_dataset("train")
        valid_clean = self.repo.load_clean_dataset("valid")

        train_texts = [s.get("Sentence", "") for s in train_clean]
        train_labels = [s.get("Emotion", "Other") for s in train_clean]

        valid_texts = [s.get("Sentence", "") for s in valid_clean]
        valid_labels = [s.get("Emotion", "Other") for s in valid_clean]

        y_train = LabelEncoderHelper.encode_labels(train_labels)
        y_val = LabelEncoderHelper.encode_labels(valid_labels)

        # Fit TF-IDF trên Train
        X_train = self.vectorizer.fit_transform(train_texts)
        X_val = self.vectorizer.transform(valid_texts)

        tuning_results: Dict[str, Any] = {}
        summary_for_heatmap: Dict[str, Dict[str, float]] = {}

        # -------------------------------------------------------------
        # Model 1: Multinomial Naive Bayes (alpha tuning)
        # -------------------------------------------------------------
        logger_model.info("Tuning Multinomial Naive Bayes...")
        nb_alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
        nb_train_f1: List[float] = []
        nb_val_f1: List[float] = []

        best_nb_score = -1.0

        for a in nb_alphas:
            nb = MultinomialNBClassifier(alpha=a)
            t0 = time.time()
            nb.train(X_train, y_train)
            t_train = time.time() - t0

            tr_pred = nb.predict(X_train)
            val_pred = nb.predict(X_val)
            val_scores = nb.predict_scores(X_val)

            tr_res = ModelEvaluator.evaluate(y_train, tr_pred, labels=ORDERED_EMOTIONS)
            val_res = ModelEvaluator.evaluate(
                y_val, val_pred, val_scores, labels=ORDERED_EMOTIONS, train_time_sec=t_train
            )

            nb_train_f1.append(tr_res["macro_f1"])
            nb_val_f1.append(val_res["macro_f1"])

            if val_res["macro_f1"] > best_nb_score:
                best_nb_score = val_res["macro_f1"]
                tuning_results["MultinomialNB"] = {"best_param": {"alpha": a}, "val_metrics": val_res}

        summary_for_heatmap["Naive Bayes"] = {
            "train_macro_f1": float(np.max(nb_train_f1)),
            "val_macro_f1": best_nb_score,
            "val_accuracy": tuning_results["MultinomialNB"]["val_metrics"]["accuracy"],
            "train_time_sec": tuning_results["MultinomialNB"]["val_metrics"]["train_time_sec"],
        }

        p_nb = tune_dir / "01_naive_bayes_alpha_tuning.png"
        TuningVisualizer.plot_hyperparameter_curve(
            param_name="alpha",
            param_values=nb_alphas,
            train_scores=nb_train_f1,
            val_scores=nb_val_f1,
            model_name="Multinomial Naive Bayes",
            output_path=p_nb,
        )

        # -------------------------------------------------------------
        # Model 2: Logistic Regression (C tuning)
        # -------------------------------------------------------------
        logger_model.info("Tuning Logistic Regression...")
        lr_c_list = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        lr_train_f1: List[float] = []
        lr_val_f1: List[float] = []

        best_lr_score = -1.0

        for c_val in lr_c_list:
            lr = LogisticRegressionClassifier(C=c_val)
            t0 = time.time()
            lr.train(X_train, y_train)
            t_train = time.time() - t0

            tr_pred = lr.predict(X_train)
            val_pred = lr.predict(X_val)
            val_scores = lr.predict_scores(X_val)

            tr_res = ModelEvaluator.evaluate(y_train, tr_pred, labels=ORDERED_EMOTIONS)
            val_res = ModelEvaluator.evaluate(
                y_val, val_pred, val_scores, labels=ORDERED_EMOTIONS, train_time_sec=t_train
            )

            lr_train_f1.append(tr_res["macro_f1"])
            lr_val_f1.append(val_res["macro_f1"])

            if val_res["macro_f1"] > best_lr_score:
                best_lr_score = val_res["macro_f1"]
                tuning_results["LogisticRegression"] = {"best_param": {"C": c_val}, "val_metrics": val_res}

        summary_for_heatmap["Logistic Regression"] = {
            "train_macro_f1": float(np.max(lr_train_f1)),
            "val_macro_f1": best_lr_score,
            "val_accuracy": tuning_results["LogisticRegression"]["val_metrics"]["accuracy"],
            "train_time_sec": tuning_results["LogisticRegression"]["val_metrics"]["train_time_sec"],
        }

        p_lr = tune_dir / "02_logistic_regression_c_tuning.png"
        TuningVisualizer.plot_hyperparameter_curve(
            param_name="C",
            param_values=lr_c_list,
            train_scores=lr_train_f1,
            val_scores=lr_val_f1,
            model_name="Logistic Regression",
            output_path=p_lr,
        )

        # -------------------------------------------------------------
        # Model 3: LinearSVC (Calibrated) (C tuning)
        # -------------------------------------------------------------
        logger_model.info("Tuning LinearSVC (Calibrated)...")
        svc_c_list = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        svc_train_f1: List[float] = []
        svc_val_f1: List[float] = []

        best_svc_score = -1.0

        for c_val in svc_c_list:
            svc = LinearSVCClassifier(C=c_val, cv=3)
            t0 = time.time()
            svc.train(X_train, y_train)
            t_train = time.time() - t0

            tr_pred = svc.predict(X_train)
            val_pred = svc.predict(X_val)
            val_scores = svc.predict_scores(X_val)

            tr_res = ModelEvaluator.evaluate(y_train, tr_pred, labels=ORDERED_EMOTIONS)
            val_res = ModelEvaluator.evaluate(
                y_val, val_pred, val_scores, labels=ORDERED_EMOTIONS, train_time_sec=t_train
            )

            svc_train_f1.append(tr_res["macro_f1"])
            svc_val_f1.append(val_res["macro_f1"])

            if val_res["macro_f1"] > best_svc_score:
                best_svc_score = val_res["macro_f1"]
                tuning_results["LinearSVC"] = {"best_param": {"C": c_val}, "val_metrics": val_res}

        summary_for_heatmap["Linear SVM (Calibrated)"] = {
            "train_macro_f1": float(np.max(svc_train_f1)),
            "val_macro_f1": best_svc_score,
            "val_accuracy": tuning_results["LinearSVC"]["val_metrics"]["accuracy"],
            "train_time_sec": tuning_results["LinearSVC"]["val_metrics"]["train_time_sec"],
        }

        p_svc = tune_dir / "03_linear_svc_c_tuning.png"
        TuningVisualizer.plot_hyperparameter_curve(
            param_name="C",
            param_values=svc_c_list,
            train_scores=svc_train_f1,
            val_scores=svc_val_f1,
            model_name="Linear SVM (Calibrated)",
            output_path=p_svc,
        )

        # -------------------------------------------------------------
        # Model 4: Decision Tree (max_depth tuning)
        # -------------------------------------------------------------
        logger_model.info("Tuning Decision Tree...")
        dt_depths = [10, 15, 20, 25, 30]
        dt_train_f1: List[float] = []
        dt_val_f1: List[float] = []

        best_dt_score = -1.0

        for d in dt_depths:
            dt = DecisionTreeModel(max_depth=d)
            t0 = time.time()
            dt.train(X_train, y_train)
            t_train = time.time() - t0

            tr_pred = dt.predict(X_train)
            val_pred = dt.predict(X_val)
            val_scores = dt.predict_scores(X_val)

            tr_res = ModelEvaluator.evaluate(y_train, tr_pred, labels=ORDERED_EMOTIONS)
            val_res = ModelEvaluator.evaluate(
                y_val, val_pred, val_scores, labels=ORDERED_EMOTIONS, train_time_sec=t_train
            )

            dt_train_f1.append(tr_res["macro_f1"])
            dt_val_f1.append(val_res["macro_f1"])

            if val_res["macro_f1"] > best_dt_score:
                best_dt_score = val_res["macro_f1"]
                tuning_results["DecisionTree"] = {"best_param": {"max_depth": d}, "val_metrics": val_res}

        summary_for_heatmap["Decision Tree"] = {
            "train_macro_f1": float(np.max(dt_train_f1)),
            "val_macro_f1": best_dt_score,
            "val_accuracy": tuning_results["DecisionTree"]["val_metrics"]["accuracy"],
            "train_time_sec": tuning_results["DecisionTree"]["val_metrics"]["train_time_sec"],
        }

        p_dt = tune_dir / "04_decision_tree_depth_tuning.png"
        TuningVisualizer.plot_hyperparameter_curve(
            param_name="max_depth",
            param_values=dt_depths,
            train_scores=dt_train_f1,
            val_scores=dt_val_f1,
            model_name="Decision Tree",
            output_path=p_dt,
        )

        # -------------------------------------------------------------
        # Model 5: Random Forest (n_estimators tuning)
        # -------------------------------------------------------------
        logger_model.info("Tuning Random Forest...")
        rf_estimators = [50, 100, 150]
        rf_train_f1: List[float] = []
        rf_val_f1: List[float] = []

        best_rf_score = -1.0

        for n_est in rf_estimators:
            rf = RandomForestModel(n_estimators=n_est, max_depth=25)
            t0 = time.time()
            rf.train(X_train, y_train)
            t_train = time.time() - t0

            tr_pred = rf.predict(X_train)
            val_pred = rf.predict(X_val)
            val_scores = rf.predict_scores(X_val)

            tr_res = ModelEvaluator.evaluate(y_train, tr_pred, labels=ORDERED_EMOTIONS)
            val_res = ModelEvaluator.evaluate(
                y_val, val_pred, val_scores, labels=ORDERED_EMOTIONS, train_time_sec=t_train
            )

            rf_train_f1.append(tr_res["macro_f1"])
            rf_val_f1.append(val_res["macro_f1"])

            if val_res["macro_f1"] > best_rf_score:
                best_rf_score = val_res["macro_f1"]
                tuning_results["RandomForest"] = {"best_param": {"n_estimators": n_est}, "val_metrics": val_res}

        summary_for_heatmap["Random Forest"] = {
            "train_macro_f1": float(np.max(rf_train_f1)),
            "val_macro_f1": best_rf_score,
            "val_accuracy": tuning_results["RandomForest"]["val_metrics"]["accuracy"],
            "train_time_sec": tuning_results["RandomForest"]["val_metrics"]["train_time_sec"],
        }

        p_rf = tune_dir / "05_random_forest_estimators_tuning.png"
        TuningVisualizer.plot_hyperparameter_curve(
            param_name="n_estimators",
            param_values=rf_estimators,
            train_scores=rf_train_f1,
            val_scores=rf_val_f1,
            model_name="Random Forest",
            output_path=p_rf,
        )

        # 6. Heatmap tổng hợp
        p_heat = tune_dir / "06_all_models_tuning_summary_heatmap.png"
        TuningVisualizer.plot_all_models_tuning_summary_heatmap(summary_for_heatmap, p_heat)

        # Lưu kết quả tuning ra JSON
        tuning_json_path = self.docs_dir / "tuning_experiments_results.json"
        with open(tuning_json_path, "w", encoding="utf-8") as f:
            clean_res = {
                k: {
                    "best_param": v["best_param"],
                    "macro_f1": v["val_metrics"]["macro_f1"],
                    "accuracy": v["val_metrics"]["accuracy"],
                    "train_time_sec": v["val_metrics"]["train_time_sec"],
                }
                for k, v in tuning_results.items()
            }
            json.dump(clean_res, f, ensure_ascii=False, indent=2)

        logger_model.info(f"Hoàn thành tuning cả 5 mô hình. Kết quả lưu tại: {tuning_json_path}")
        return tuning_results

    # =========================================================================
    # 4. ĐÁNH GIÁ ĐỐI ĐẦU BEST MODELS & LƯU BEST MODEL (docs/figures/model_comparison/)
    # =========================================================================
    def evaluate_and_save_best_models(self) -> Dict[str, Any]:
        """Đánh giá 5 mô hình tối ưu trên tập Test, vẽ 5 biểu đồ đối đầu và lưu best_model.joblib."""
        logger_model.info("=== BẮT ĐẦU ĐỐI ĐẦU 5 BEST MODELS TRÊN TẬP TEST ===")
        comp_dir = self.figures_dir / "model_comparison"
        comp_dir.mkdir(parents=True, exist_ok=True)

        train_clean = self.repo.load_clean_dataset("train")
        test_clean = self.repo.load_clean_dataset("test")

        train_texts = [s.get("Sentence", "") for s in train_clean]
        train_labels = [s.get("Emotion", "Other") for s in train_clean]

        test_texts = [s.get("Sentence", "") for s in test_clean]
        test_labels = [s.get("Emotion", "Other") for s in test_clean]

        y_train = LabelEncoderHelper.encode_labels(train_labels)
        y_test = LabelEncoderHelper.encode_labels(test_labels)

        # Vectorize
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)

        # Đọc tham số tốt nhất từ tuning nếu có, hoặc dùng tham số chuẩn
        tuning_path = self.docs_dir / "tuning_experiments_results.json"
        best_params: Dict[str, Any] = {}
        if tuning_path.exists():
            with open(tuning_path, "r", encoding="utf-8") as f:
                best_params = json.load(f)

        nb_alpha = best_params.get("MultinomialNB", {}).get("best_param", {}).get("alpha", 0.5)
        lr_c = best_params.get("LogisticRegression", {}).get("best_param", {}).get("C", 1.0)
        svc_c = best_params.get("LinearSVC", {}).get("best_param", {}).get("C", 1.0)
        dt_depth = best_params.get("DecisionTree", {}).get("best_param", {}).get("max_depth", 20)
        rf_n = best_params.get("RandomForest", {}).get("best_param", {}).get("n_estimators", 100)

        # Khởi tạo 5 Best Candidates
        candidates = {
            "Naive Bayes": MultinomialNBClassifier(alpha=nb_alpha),
            "Logistic Regression": LogisticRegressionClassifier(C=lr_c),
            "Linear SVM (Calibrated)": LinearSVCClassifier(C=svc_c, cv=3),
            "Decision Tree": DecisionTreeModel(max_depth=dt_depth),
            "Random Forest": RandomForestModel(n_estimators=rf_n, max_depth=25),
        }

        test_results: Dict[str, Dict[str, Any]] = {}
        all_y_preds: Dict[str, np.ndarray] = {}
        all_y_scores: Dict[str, np.ndarray] = {}

        best_overall_model = None
        best_overall_name = ""
        best_overall_macro_f1 = -1.0

        for name, model in candidates.items():
            logger_model.info(f"Huấn luyện và đánh giá trên tập Test: {name}...")
            t0 = time.time()
            model.train(X_train, y_train)
            train_time = time.time() - t0

            y_pred = model.predict(X_test)
            y_scores = model.predict_scores(X_test)

            eval_res = ModelEvaluator.evaluate(
                y_true=y_test,
                y_pred=y_pred,
                y_scores=y_scores,
                labels=ORDERED_EMOTIONS,
                train_time_sec=train_time,
            )

            test_results[name] = eval_res
            all_y_preds[name] = y_pred
            all_y_scores[name] = y_scores

            if eval_res["macro_f1"] > best_overall_macro_f1:
                best_overall_macro_f1 = eval_res["macro_f1"]
                best_overall_name = name
                best_overall_model = model

        logger_model.info(f"🏆 MÔ HÌNH VÔ ĐỊCH: {best_overall_name} (Macro-F1: {best_overall_macro_f1:.4f})")

        # -------------------------------------------------------------
        # Sinh 5 Biểu đồ Đối đầu (docs/figures/model_comparison/)
        # -------------------------------------------------------------
        # 1. Bar Chart so sánh Macro-F1, Weighted-F1, Accuracy
        p1 = comp_dir / "01_best_models_benchmark_bar.png"
        BenchmarkVisualizer.plot_best_models_benchmark(test_results, p1)

        # 2. Scatter plot Training Time vs Macro-F1 Trade-off
        p2 = comp_dir / "02_best_models_training_latency_tradeoff.png"
        BenchmarkVisualizer.plot_training_latency_tradeoff(test_results, p2)

        # 3. Confusion Matrix Best vs Baseline (Naive Bayes)
        p3 = comp_dir / "03_confusion_matrix_best_vs_baseline.png"
        BenchmarkVisualizer.plot_confusion_matrix_comparison(
            cm_best=test_results[best_overall_name]["confusion_matrix"],
            cm_baseline=test_results["Naive Bayes"]["confusion_matrix"],
            labels=ORDERED_EMOTIONS,
            best_name=best_overall_name,
            baseline_name="Naive Bayes",
            output_path=p3,
        )

        # 4. Multiclass ROC Curves cho Best Model
        p4 = comp_dir / "04_multiclass_roc_curves_best_model.png"
        BenchmarkVisualizer.plot_multiclass_roc_curves(
            roc_data=test_results[best_overall_name]["roc_data"],
            model_name=best_overall_name,
            output_path=p4,
        )

        # 5. Radar Chart so sánh F1 7 nhãn
        models_per_class_f1 = {m: test_results[m]["per_class_f1"] for m in test_results}
        p5 = comp_dir / "05_per_class_f1_radar_chart.png"
        BenchmarkVisualizer.plot_per_class_f1_radar(models_per_class_f1, ORDERED_EMOTIONS, p5)

        # -------------------------------------------------------------
        # Lưu trữ Best Model & TF-IDF Vectorizer
        # -------------------------------------------------------------
        best_model_path = self.saved_models_dir / "best_model.joblib"
        tfidf_path = self.saved_models_dir / "tfidf_vectorizer.joblib"
        mapping_path = self.saved_models_dir / "label_mapping.json"

        if best_overall_model is not None:
            best_overall_model.save(best_model_path)
        self.vectorizer.save(tfidf_path)
        LabelEncoderHelper.export_mapping(mapping_path)

        # Xuất file báo cáo số liệu JSON
        benchmark_report_path = self.docs_dir / "model_benchmark_results.json"
        clean_report: Dict[str, Any] = {
            "best_model_name": best_overall_name,
            "best_macro_f1": best_overall_macro_f1,
            "models": {
                m: {
                    "accuracy": test_results[m]["accuracy"],
                    "macro_f1": test_results[m]["macro_f1"],
                    "weighted_f1": test_results[m]["weighted_f1"],
                    "log_loss": test_results[m]["log_loss"],
                    "brier_score": test_results[m]["brier_score"],
                    "train_time_sec": test_results[m]["train_time_sec"],
                    "per_class_f1": test_results[m]["per_class_f1"],
                }
                for m in test_results
            },
        }

        with open(benchmark_report_path, "w", encoding="utf-8") as f:
            json.dump(clean_report, f, ensure_ascii=False, indent=2)

        logger_model.info(f"Đã lưu Best Model vào: {best_model_path}")
        logger_model.info(f"Đã xuất báo cáo Benchmark vào: {benchmark_report_path}")

        return clean_report

    # =========================================================================
    # 5. THỰC THI TOÀN BỘ ĐƯỜNG ỐNG (RUN ALL A-Z)
    # =========================================================================
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Thực hiện toàn bộ đường ống tuần tự: Làm sạch -> EDA -> Tuning -> Đối đầu."""
        logger_pipe.info("🚀 BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE TỰ ĐỘNG TỪ ĐẦU ĐẾN CUỐI")
        counts = self.clean_and_save_data()
        eda_figs = self.generate_eda_charts()
        tuning_res = self.run_hyperparameter_tuning()
        bench_res = self.evaluate_and_save_best_models()

        return {
            "clean_counts": counts,
            "eda_figures": [str(p) for p in eda_figs],
            "tuning_models": list(tuning_res.keys()),
            "best_model": bench_res["best_model_name"],
            "best_macro_f1": bench_res["best_macro_f1"],
        }
