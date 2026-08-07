from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def table3_comparison(root: Path, results: Path) -> Path | None:
    local = rows(results / "table3_local.csv")
    paper = {row["method"]: row for row in rows(root / "data/paper/table3.csv")}
    if not local:
        return None
    metrics = ["fluency", "identification", "comforting", "suggestion", "overall"]
    output = results / "table3_comparison.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        fields = ["method", "n"] + [part for m in metrics for part in (f"paper_{m}", f"local_{m}", f"abs_diff_{m}")]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in local:
            source = paper.get(row["method"], {})
            out = {"method": row["method"], "n": row.get("n", "")}
            for metric in metrics:
                p = float(source[metric]) if source.get(metric) else None
                value = float(row[metric])
                out[f"paper_{metric}"] = "" if p is None else f"{p:.4f}"
                out[f"local_{metric}"] = f"{value:.4f}"
                out[f"abs_diff_{metric}"] = "" if p is None else f"{abs(value-p):.4f}"
            writer.writerow(out)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    comparison3 = table3_comparison(root, args.results)
    statuses = [
        ("Table 1 自动指标", args.results / "table1_local.csv", args.results / "table1_comparison.csv", "真实生成完成后可得到"),
        ("Table 2 三人盲评", args.results / "table2_local.csv", args.results / "table2_blind_annotation_form.csv", "必须由3名真人完成A/B/TIE标注"),
        ("Table 3 本地LLM裁判", args.results / "table3_local.csv", comparison3, "本地模型替代GPT-4o，属于近似再评估"),
        ("Table 4 消融", args.results / "table4_local_and_comparison.csv", None, "消融为论文描述下的近似再实现"),
    ]
    lines = [
        "# MultiAgentESC Table 1–4 本机复现状态报告",
        "",
        "> 本报告只把真实模型调用或真人标注产生的结果记为本机结果；mock测试不会进入报告。",
        "",
        "| 表格 | 状态 | 本机结果 | 对照/任务文件 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for name, local, aux, note in statuses:
        status = "已完成" if local.exists() else "待完成"
        local_text = str(local) if local.exists() else "—"
        aux_text = str(aux) if aux and aux.exists() else "—"
        lines.append(f"| {name} | {status} | `{local_text}` | `{aux_text}` | {note} |")
    lines += [
        "",
        "## 复现声明",
        "",
        "- Zero-shot、Few-shot、Zero-shot CoT、Few-shot CoT采用论文附录公开提示词。",
        "- 其余baseline依据论文描述补写；论文未公开逐样本生成代码，因此属于近似再实现。",
        "- MultiAgentESC与公开流程、提示结构和关键开关对齐，但为便于统一实验而重构，不等同于作者原仓库逐行执行。",
        "- 自动指标为透明代理实现；若作者tokenizer或平滑方式不同，数值可能存在系统偏差。",
        "- Table 2没有真人标注时不得声称完成；Table 3使用本地LLM替代GPT-4o，只能称为近似再评估。",
    ]
    output = args.results / "FINAL_STATUS_REPORT_ZH.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
