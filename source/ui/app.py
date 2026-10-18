"""Streamlit Web UI for Vietnamese Sentiment Analysis (UIT-VSMEC).

Clean Architecture: Tầng Presentation điều phối toàn bộ qua SentimentPipelineOrchestrator.
"""

from typing import Dict

import pandas as pd
import streamlit as st

from source.domain.entities.sentiment_sample import EmotionLabel
from source.infrastructure.interact.visualization.emotion_charts import EmotionVisualizer
from source.services.cli import SentimentPipelineOrchestrator

# Thiết lập cấu hình trang Streamlit
st.set_page_config(
    page_title="Phân loại cảm xúc tiếng Việt (UIT-VSMEC)",
    page_icon="🎭",
    layout="wide",
)

st.title("🎭 Hệ thống Phân loại Cảm xúc Văn bản Tiếng Việt")
st.caption("Đề tài môn học IT3190 - Nhập môn Học máy & Khai phá dữ liệu | Nhóm: Hải Nam")

# Khởi tạo bộ điều phối chính (nạp mô hình tốt nhất nếu có)
orchestrator = SentimentPipelineOrchestrator.build_default()

# Cột bên trái: Nhập văn bản & Phân tích
col_input, col_stats = st.columns([1.2, 0.8])

with col_input:
    st.subheader("📝 Nhập câu văn bản")
    user_text = st.text_area(
        label="Nội dung cần dự đoán cảm xúc:",
        value="Hôm nay hoàn thành bài tập lớn vui quá xá!",
        height=100,
        help="Nhập bất kỳ câu tiếng Việt nào, có thể kèm teencode hoặc emoji.",
    )

    analyze_clicked = st.button("🚀 Phân tích cảm xúc", type="primary", use_container_width=True)

    if analyze_clicked and user_text.strip():
        # Gọi qua Bộ điều phối chính duy nhất của hệ thống
        profile = orchestrator.analyze_text(user_text)

        st.divider()
        st.subheader("🎯 Thang đo Phổ điểm 7 Chiều Cảm xúc")

        badge_labels: Dict[EmotionLabel, str] = {
            EmotionLabel.ENJOYMENT: "🟢 Thích thú (Enjoyment)",
            EmotionLabel.SADNESS: "🔵 Buồn bã (Sadness)",
            EmotionLabel.ANGER: "🔴 Giận dữ (Anger)",
            EmotionLabel.FEAR: "🟣 Sợ hãi (Fear)",
            EmotionLabel.DISGUST: "🟠 Ghê tởm (Disgust)",
            EmotionLabel.SURPRISE: "🟡 Bất ngờ (Surprise)",
            EmotionLabel.OTHER: "⚪ Khác (Other)",
        }

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            dom_text = badge_labels.get(profile.dominant_emotion, profile.dominant_emotion.value)
            st.success(f"**Cảm xúc chủ đạo:** {dom_text}")
        with col_metric2:
            st.metric(label="Cường độ cực đại (Intensity)", value=f"{profile.intensity * 100:.1f}%")

        if profile.is_mixed:
            st.warning(
                "⚠️ **Phát hiện cảm xúc pha trộn (Mixed Emotions):** "
                "Câu văn đồng thời xuất hiện sắc thái cảm xúc thứ hai khá rõ nét!"
            )

        # Tabs hiển thị 2 dạng đồ thị trực quan hóa
        tab_radar, tab_bar, tab_data = st.tabs(
            ["🕸️ Biểu đồ Mạng nhện (Radar)", "📊 Thanh đo Quang phổ", "📋 Bảng số liệu"]
        )

        with tab_radar:
            st.caption("Biểu đồ mạng nhện biểu diễn đa giác vùng phủ của 7 cảm xúc trên hệ tọa độ cực:")
            fig_radar = EmotionVisualizer.plot_radar_chart(profile)
            st.pyplot(fig_radar)

        with tab_bar:
            st.caption("Thanh đo cường độ từng sắc thái cảm xúc:")
            fig_bar = EmotionVisualizer.plot_spectrum_bar(profile)
            st.pyplot(fig_bar)

        with tab_data:
            df_scores = pd.DataFrame(
                [{"Cảm xúc": lbl.value, "Điểm số": f"{score * 100:.1f}%"} for lbl, score in profile.ranking]
            )
            st.dataframe(df_scores, use_container_width=True)

# Cột bên phải: Thông tin tập dữ liệu & Thống kê
with col_stats:
    st.subheader("📚 Thông tin UIT-VSMEC")
    st.info(
        """
        - **Dữ liệu:** Vietnamese Social Media Emotion Corpus (~6,927 câu).
        - **7 nhãn:** Enjoyment, Sadness, Anger, Fear, Disgust, Surprise, Other.
        - **Đầu ra:** Thang đo phổ điểm 7 chiều $[S_1 .. S_7] \\in [0, 1]$.
        - **Kiến trúc:** Clean Architecture (Domain $\\rightarrow$ Services/CLI $\\rightarrow$ Presentation).
        """
    )
    st.markdown("### 🏆 Bảng phân bổ các nhãn cảm xúc")
    demo_distribution = pd.DataFrame(
        {
            "Số mẫu ước tính": [1971, 952, 597, 394, 381, 252, 2380],
        },
        index=[
            "Enjoyment",
            "Sadness",
            "Anger",
            "Fear",
            "Disgust",
            "Surprise",
            "Other",
        ],
    )
    st.dataframe(demo_distribution, use_container_width=True)
