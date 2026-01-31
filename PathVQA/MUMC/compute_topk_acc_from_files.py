#!/usr/bin/env python3
"""
根据top-k结果与compare文件计算Top-1/3/5准确率
用于核对all_acc_topk.json中的指标
"""
import json
import argparse
from dataset.utils import pre_answer


def ans_type(answer: str) -> str:
    a = answer.strip().lower()
    if a in {"yes", "no"}:
        return "yes/no"
    try:
        float(a)
        return "number"
    except Exception:
        return "other"


def percent(n: int, d: int) -> float:
    return round(100.0 * n / max(d, 1), 2)


def compute_metrics(topk_path: str, compare_path: str):
    # 读取top-k结果（只保留前5个候选答案及概率）
    with open(topk_path, 'r') as f:
        topk_data = json.load(f)

    qid_to_preds = {}
    for item in topk_data:
        qid = int(item["qid"])
        entries = item.get("answers", [])
        normalized = []
        for entry in entries[:5]:
            if isinstance(entry, dict):
                ans_text = entry.get("answer", "")
            else:
                ans_text = entry
            normalized.append(pre_answer(ans_text))
        qid_to_preds[qid] = normalized

    # 读取ground truth；compare文件格式为 {qid: [pred, gt]}
    with open(compare_path, 'r') as f:
        compare_data = json.load(f)
    qid_to_gt = {int(k): pre_answer(v[0]) for k, v in compare_data.items()}

    # 计算Top-1/3/5准确率
    rows = []
    for k in [1, 3, 5]:
        totals = {"overall": 0, "yes/no": 0, "number": 0, "other": 0}
        hits = {"overall": 0, "yes/no": 0, "number": 0, "other": 0}

        for qid, gt in qid_to_gt.items():
            preds = qid_to_preds.get(qid, [])
            atype = ans_type(gt)

            totals["overall"] += 1
            totals[atype] += 1

            if gt in preds[:k]:
                hits["overall"] += 1
                hits[atype] += 1

        rows.append({
            "TopK": k,
            "Overall": percent(hits["overall"], totals["overall"]),
            "yes/no": percent(hits["yes/no"], totals["yes/no"]),
            "number": percent(hits["number"], totals["number"]),
            "other": percent(hits["other"], totals["other"]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk_path", required=True, help="路径：*_vqa_result_0_topk.json")
    parser.add_argument("--compare_path", required=True, help="路径：*_vqa_result_0_compare.json")
    args = parser.parse_args()

    metrics = compute_metrics(args.topk_path, args.compare_path)
    print("Top-k accuracy based on provided files:")
    for row in metrics:
        print(f"Top{row['TopK']}: Overall={row['Overall']}%, "
              f"Yes/No={row['yes/no']}%, Number={row['number']}%, Other={row['other']}%")


if __name__ == "__main__":
    main()

