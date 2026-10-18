import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.services.cli.orchestrator import SentimentPipelineOrchestrator

QUERIES = [
    {
        "id": 1,
        "text": "khi nào phường có chế độ phát chồng khi mang thân đến đăng kí kết hôn tự nguyện 😊",
        "expected": "Enjoyment",
    },
    {
        "id": 2,
        "text": "làm công nhân đã bị vắt kiệt sức khỏe cho đến khi bị thải",
        "expected": "Sadness",
    },
    {
        "id": 3,
        "text": "thái độ phục vụ quá mất dạy, lừa đảo khách hàng coi chừng quả báo",
        "expected": "Anger",
    },
    {
        "id": 4,
        "text": "đi đêm một mình qua đoạn đường vắng này sợ toát cả mồ hôi, rùng mình",
        "expected": "Fear",
    },
    {
        "id": 5,
        "text": "nhìn thấy bát bún có giòi bò lúc nhúc tởm lợm kinh khủng khiếp",
        "expected": "Disgust",
    },
    {
        "id": 6,
        "text": "thật không thể tin nổi là mình lại trúng số độc đắc ối giời ơi",
        "expected": "Surprise",
    },
    {
        "id": 7,
        "text": "hôm nay ngày mai thời tiết bình thường, đi học lúc 7 giờ sáng",
        "expected": "Other",
    },
    {
        "id": 8,
        "text": "ôi bất ngờ quá, món quà sinh nhật xịn sò tuyệt cú mèo luôn 🎁😍",
        "expected": "Enjoyment",
    },
    {
        "id": 9,
        "text": "haizzz ... thế là chưa có kq gì tốt đẹp cả .. bản thân cố gắng cũng chẳng dễ dàng gì :((",
        "expected": "Sadness",
    },
    {
        "id": 10,
        "text": "phải bồi thường thiệt hại bị trầy xước thân tàu . tiền sơn lại mấy toa tàu chắc cũng bộn .",
        "expected": "Other",
    },
]

def run_eval():
    orchestrator = SentimentPipelineOrchestrator.build_default()
    has_classifier = orchestrator.classifier is not None
    print(f"[*] Orchestrator classifier loaded: {has_classifier}")
    if has_classifier:
        print(f"[*] Model type: {type(orchestrator.classifier).__name__}")
    
    results = []
    balanced_count = 0

    for item in QUERIES:
        profile = orchestrator.analyze_text(item["text"])
        scores = {k: round(v, 4) for k, v in profile.scores.items()}
        
        # Kiểm tra xem có bị cân bằng 14.3% không
        is_balanced = all(abs(v - 0.143) < 0.01 for v in scores.values())
        if is_balanced:
            balanced_count += 1

        top_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        res = {
            "id": item["id"],
            "text": item["text"],
            "expected": item["expected"],
            "predicted": profile.dominant_emotion.value,
            "intensity": round(profile.intensity, 4),
            "is_mixed": profile.is_mixed,
            "is_balanced_fallback": is_balanced,
            "top_3_emotions": top_emotions,
            "all_scores": scores,
        }
        results.append(res)
        print(f"[{item['id']}] Text: {item['text'][:45]}...")
        print(f"    Expected: {item['expected']} | Predicted: {profile.dominant_emotion.value} (Intensity: {profile.intensity*100:.1f}%) | Balanced: {is_balanced}")
        print(f"    Top 3: {top_emotions}\n")

    print(f"[*] Summary: {len(QUERIES) - balanced_count}/{len(QUERIES)} non-balanced predictions (Balanced fallbacks: {balanced_count})")
    
    with open("docs/10_queries_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_eval()
