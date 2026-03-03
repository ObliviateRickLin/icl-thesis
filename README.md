# moe-icl (src-focused)

This branch is cleaned to keep `src/` as the primary codebase.

## Kept
- `src/` core GPT/ICL training + evaluation + SpeedCP/CondConf pipeline
- `src/conf/gpt/` configs
- shared config files used by GPT flow

## Removed
- `scaling_report/`
- non-GPT config families under `src/conf/` (`encoder`, `gemma`, `llama`, `qwen`)
- clearly unrelated encoder/routing analysis scripts

If you want, I can continue and do a second pass inside `src/` to further prune old helpers and one-off scripts.