import os
import json
import hashlib
import fcntl
from pathlib import Path

import torch
import tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path("/root/autodl-tmp/paper_repro")
MODEL = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

DATASETS = [
    {
        "name": "PersonaMem-128K",
        "dir": ROOT / "adapters/scm_personamem_128k_dedup",
        "expected": 12841,
        "needed": 9663,
        "short": 3178,
    },
    {
        "name": "PersonaMem-1M",
        "dir": ROOT / "adapters/scm_personamem_1m",
        "expected": 23502,
        "needed": 18736,
        "short": 4766,
    },
]

PREFIX = (
    "Below is a conversation between a user and an AI assistant. "
    "Please write a summary for each of them in one sentence and "
    "list them in separate paragraphs, while trying to preserve "
    "the key information of the user’s question and the assistant’s "
    "answer as much as possible.\n\n"
    "conversation content: \n\n"
)

SUFFIX = "\n\nSummary:\n"
THRESHOLD = 300


def load_jsonl(path):
    rows = []
    if not path.exists():
        return rows

    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(
                    f"JSON损坏: {path}:{n}"
                ) from e
    return rows


def atomic_json(obj, path):
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():

    lock_path = ROOT / "adapters/scm_personamem/.summary_multi.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock = open(lock_path, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("已有摘要生成任务运行，拒绝重复启动")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")

    if not Path(MODEL).exists():
        raise RuntimeError(f"模型不存在: {MODEL}")

    try:
        enc = tiktoken.encoding_for_model("text-davinci-003")
    except Exception:
        enc = tiktoken.get_encoding("p50k_base")

    prepared = []

    # =====================================================
    # 1. CPU预检、统计token、补齐short path
    # =====================================================

    for cfg in DATASETS:

        name = cfg["name"]
        base = cfg["dir"]

        print(f"\n===== PREFLIGHT {name} =====", flush=True)

        unique_path = base / "unique_scm_memory_units.jsonl"
        output_path = base / "scm_summaries_qwen7b.jsonl"
        audit_path = base / "scm_memory_unit_audit.json"

        units = load_jsonl(unique_path)

        if len(units) != cfg["expected"]:
            raise RuntimeError(
                f"{name}: unique数量错误 {len(units)}"
            )

        audit = json.load(open(audit_path, encoding="utf-8"))

        if audit["summary_needed_unique"] != cfg["needed"]:
            raise RuntimeError(f"{name}: summary_needed不一致")

        if audit["summary_short_path_unique"] != cfg["short"]:
            raise RuntimeError(f"{name}: short_path不一致")

        unit_map = {}

        for x in units:

            h = x["unit_hash"]
            raw = x["raw_text"]

            actual = hashlib.sha256(
                raw.encode("utf-8")
            ).hexdigest()

            if actual != h:
                raise RuntimeError(f"SHA256错误: {h}")

            if h in unit_map:
                raise RuntimeError(f"重复unit: {h}")

            n = len(enc.encode(PREFIX + raw + SUFFIX))

            unit_map[h] = {
                "raw": raw,
                "tokens": n,
                "needs": n > THRESHOLD,
            }

        needed = sum(x["needs"] for x in unit_map.values())

        if needed != cfg["needed"]:
            raise RuntimeError(
                f"{name}: 实际需要生成{needed}，"
                f"预期{cfg['needed']}"
            )

        done = {}

        for x in load_jsonl(output_path):

            h = x["unit_hash"]

            if h not in unit_map:
                raise RuntimeError(f"输出包含未知hash: {h}")

            if h in done:
                raise RuntimeError(f"输出存在重复hash: {h}")

            if not isinstance(x.get("summary"), str) or not x["summary"].strip():
                raise RuntimeError(f"空摘要: {h}")

            expected_source = (
                "qwen2.5-7b"
                if unit_map[h]["needs"]
                else "raw_under_300"
            )

            if x.get("source") != expected_source:
                raise RuntimeError(f"摘要来源不匹配: {h}")

            if not unit_map[h]["needs"]:
                if x["summary"] != unit_map[h]["raw"]:
                    raise RuntimeError(f"short path原文不一致: {h}")

            done[h] = x

        # 补齐无需LLM生成的记录
        with output_path.open("a", encoding="utf-8") as f:

            for h, u in unit_map.items():

                if h in done or u["needs"]:
                    continue

                rec = {
                    "unit_hash": h,
                    "summary": u["raw"],
                    "source": "raw_under_300",
                    "prompt_tokens": u["tokens"],
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[h] = rec

        remaining = [
            (h, u)
            for h, u in unit_map.items()
            if u["needs"] and h not in done
        ]

        print("Unique:", len(unit_map), flush=True)
        print("Valid cache:", len(done), flush=True)
        print("Need generation:", len(remaining), flush=True)

        prepared.append({
            "cfg": cfg,
            "unit_map": unit_map,
            "done": done,
            "remaining": remaining,
            "output": output_path,
        })

    # =====================================================
    # 2. 两套数据都没有待生成任务时，不加载模型
    # =====================================================

    total_remaining = sum(
        len(x["remaining"]) for x in prepared
    )

    if total_remaining:

        print("\n===== LOAD QWEN =====", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            local_files_only=True,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={
                0: "18GiB",
                "cpu": "24GiB",
            },
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        )

        model.eval()

        print(
            "HF_DEVICE_MAP:",
            getattr(model, "hf_device_map", None),
            flush=True,
        )

        # 保留已验证过的lm_head显存优化
        def lm_head_pre_hook(module, args):
            hidden = args[0]
            if hidden.ndim == 3 and hidden.shape[1] > 1:
                hidden = hidden[:, -1:, :]
            return (hidden,)

        hook = model.lm_head.register_forward_pre_hook(
            lm_head_pre_hook
        )

        try:

            input_device = model.get_input_embeddings().weight.device

            # =================================================
            # 3. 依次生成128K、1M
            # =================================================

            for item in prepared:

                cfg = item["cfg"]
                remaining = item["remaining"]

                name = cfg["name"]
                output_path = item["output"]

                print(
                    f"\n===== GENERATE {name} =====",
                    flush=True,
                )

                with output_path.open("a", encoding="utf-8") as f:

                    for i, (h, u) in enumerate(remaining, 1):

                        prompt = (
                            PREFIX
                            + u["raw"]
                            + SUFFIX
                        )

                        text = tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )

                        inputs = tokenizer(
                            text,
                            return_tensors="pt",
                        )

                        inputs = {
                            k: v.to(input_device)
                            for k, v in inputs.items()
                        }

                        max_ctx = getattr(
                            model.config,
                            "max_position_embeddings",
                            32768,
                        )

                        if inputs["input_ids"].shape[1] > max_ctx:
                            raise RuntimeError(
                                f"输入超过模型上下文限制: {h}"
                            )

                        with torch.inference_mode():

                            output = model.generate(
                                **inputs,
                                max_new_tokens=256,
                                do_sample=False,
                                use_cache=True,
                                pad_token_id=tokenizer.eos_token_id,
                            )

                        new_ids = output[
                            0,
                            inputs["input_ids"].shape[1]:
                        ]

                        summary = tokenizer.decode(
                            new_ids,
                            skip_special_tokens=True,
                        ).strip()

                        if not summary:
                            raise RuntimeError(
                                f"生成空摘要: {h}"
                            )

                        rec = {
                            "unit_hash": h,
                            "summary": summary,
                            "source": "qwen2.5-7b",
                            "prompt_tokens": u["tokens"],
                        }

                        f.write(
                            json.dumps(
                                rec,
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        f.flush()

                        item["done"][h] = rec

                        if i == 1 or i % 25 == 0 or i == len(remaining):
                            print(
                                f"[{name}] "
                                f"{i}/{len(remaining)} "
                                f"hash={h[:12]} "
                                f"chars={len(summary)}",
                                flush=True,
                            )

        finally:
            hook.remove()
            del model
            torch.cuda.empty_cache()

    # =====================================================
    # 4. 最终完整性审计
    # =====================================================

    for item in prepared:

        cfg = item["cfg"]
        base = cfg["dir"]

        final = load_jsonl(item["output"])

        by_hash = {}

        for x in final:

            h = x["unit_hash"]

            if h in by_hash:
                raise RuntimeError(f"最终输出重复: {h}")

            if h not in item["unit_map"]:
                raise RuntimeError(f"未知hash: {h}")

            if not isinstance(x.get("summary"), str) or not x["summary"].strip():
                raise RuntimeError(f"空摘要: {h}")

            by_hash[h] = x

        expected_hashes = set(item["unit_map"])
        actual_hashes = set(by_hash)

        if expected_hashes != actual_hashes:
            raise RuntimeError(
                f"{cfg['name']} 缺少摘要: "
                f"{len(expected_hashes - actual_hashes)}"
            )

        audit = {
            "dataset": cfg["name"],
            "unique_units": len(by_hash),
            "summary_records": len(by_hash),
            "qwen_generated": sum(
                x["source"] == "qwen2.5-7b"
                for x in final
            ),
            "raw_short_path": sum(
                x["source"] == "raw_under_300"
                for x in final
            ),
            "missing": 0,
            "duplicates": 0,
            "all_summaries_nonempty": True,
            "status": "PASS",
        }

        atomic_json(
            audit,
            base / "scm_summary_generation_audit.json",
        )

        print("\n===== DATASET COMPLETE =====", flush=True)
        print(json.dumps(audit, indent=2), flush=True)

    print("\nSCM_BOTH_DATASETS_SUMMARY_OK", flush=True)


if __name__ == "__main__":
    main()
