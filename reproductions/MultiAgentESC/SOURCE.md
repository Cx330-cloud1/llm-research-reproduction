# Sources and reproducibility boundary

- Paper: https://aclanthology.org/2025.emnlp-main.232/
- Official repository: https://github.com/MindIntLab-HFUT/MultiAgentESC
- Audited public commit: `631b7f1961fc7502e547fd9258e847230dbcb973`
- Dataset bundled here: the `data/ESConv.json` file from that public commit.

The official repository publishes the MultiAgentESC main workflow, prompts, and ESConv data, but not the full baseline runners, exact automatic metric code, raw human annotations, GPT-4o raw scores/mapping, or executable ablation switches. This package therefore labels missing components as reimplementations and does not claim byte-for-byte reproduction of unpublished code.

Table 3 in this package deliberately uses a local Qwen judge through Ollama. It does not call OpenAI and must be reported as a local-LLM approximate reevaluation rather than a reproduction of the paper's GPT-4o evaluation.
