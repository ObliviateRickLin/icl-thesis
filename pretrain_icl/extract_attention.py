#!/usr/bin/env python3
"""Extract attention weights from Qwen3-8B to analyze distance-based weighting."""
import sys, torch, numpy as np, json
sys.path.insert(0, "/root/moe-icl-aris/src")
from eval_icl_llm_rmse import _seed_everything, collect_xy, _build_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

model_path = "/gemini/code/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32,
    device_map="cpu", trust_remote_code=True, attn_implementation="eager")
model.eval()
print("Model loaded")

def analyze_attention(d, L, n_episodes=5, seed=42):
    _seed_everything(seed)
    xs, ys = collect_xy(task_name="noisy_linear_regression",
        task_kwargs={"noise_std": 0.1, "normalize_w": True},
        data_name="gaussian", n_dims=d, n_points=L+1,
        num_eval_examples=n_episodes, batch_size=n_episodes, device="cpu")
    xs_np, ys_np = xs.numpy(), ys.numpy()

    all_attn_by_rank = [[] for _ in range(L)]

    for ep in range(n_episodes):
        prompt = _build_prompt(xs_ctx=xs_np[ep, :L], ys_ctx=ys_np[ep, :L], x_query=xs_np[ep, L],
            prompt_style="words2numbers", answer_format="first_number", x_decimals=2, y_decimals=2)

        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0].tolist()]

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Find y-value token positions: "Output" immediately followed by ":"
        y_positions = []
        for pos in range(len(tokens) - 1):
            if tokens[pos].strip() == "Output" and tokens[pos+1].strip() == ":":
                y_start = pos + 2
                while y_start < len(tokens) and tokens[y_start].strip() == "":
                    y_start += 1
                if y_start < len(tokens):
                    y_positions.append(y_start)

        # Last one is query Output:, keep only context y positions
        if len(y_positions) > L:
            y_positions = y_positions[-L-1:-1]  # last L+1 minus the query
        elif len(y_positions) == L + 1:
            y_positions = y_positions[:-1]
        if len(y_positions) != L:
            print("  ep%d: found %d y-positions, expected %d, skip" % (ep, len(y_positions), L))
            continue

        dists = np.sqrt(np.sum((xs_np[ep, :L] - xs_np[ep, L])**2, axis=1))
        ranks = np.argsort(dists)

        # Attention from last token to y-value positions
        last_pos = len(tokens) - 1
        n_layers = len(outputs.attentions)

        # Use last 4 layers, average across heads
        attn_to_y = np.zeros(L)
        for layer_idx in range(n_layers - 4, n_layers):
            attn = outputs.attentions[layer_idx][0].numpy()  # (n_heads, seq_len, seq_len)
            attn_avg = attn.mean(axis=0)  # (seq_len, seq_len)
            for i, ypos in enumerate(y_positions):
                attn_to_y[i] += attn_avg[last_pos, ypos]
        attn_to_y /= 4

        # Normalize
        attn_sum = attn_to_y.sum()
        if attn_sum > 0:
            attn_norm = attn_to_y / attn_sum
        else:
            continue

        # Store by rank
        for rank_idx in range(L):
            ctx_idx = ranks[rank_idx]
            all_attn_by_rank[rank_idx].append(attn_norm[ctx_idx])

        if ep == 0:
            print("  ep0 tokens=%d, y_positions=%s" % (len(tokens), y_positions[:5]))

    return all_attn_by_rank


# Run for multiple conditions and seeds
print("\n" + "=" * 80)
SEEDS = [42, 123, 456, 789, 1024]

for d, L in [(1, 10), (1, 20), (2, 12), (3, 12), (5, 20)]:
    print("\n=== d=%d, L=%d ===" % (d, L))
    all_corrs = []
    combined_by_rank = [[] for _ in range(L)]

    for seed in SEEDS:
        attn_by_rank = analyze_attention(d, L, n_episodes=10, seed=seed)

        means = [np.mean(attn_by_rank[r]) if attn_by_rank[r] else 0 for r in range(L)]
        if any(m > 0 for m in means):
            r_corr, _ = stats.pearsonr(range(L), means)
            all_corrs.append(r_corr)
            for r in range(L):
                combined_by_rank[r].extend(attn_by_rank[r])
            print("  seed=%d: corr=%.3f, rank0=%.4f, rank_last=%.4f" % (
                seed, r_corr, means[0], means[-1]))

    if all_corrs:
        print("  --- Summary: mean_corr=%.3f (std=%.3f) across %d seeds ---" % (
            np.mean(all_corrs), np.std(all_corrs), len(all_corrs)))
        print("  Combined rank profile:")
        print("  Rank  Mean_attn  Std_attn   N")
        for r in range(min(L, 10)):
            ws = combined_by_rank[r]
            if ws:
                print("  %2d    %.5f    %.5f    %d" % (r, np.mean(ws), np.std(ws), len(ws)))
