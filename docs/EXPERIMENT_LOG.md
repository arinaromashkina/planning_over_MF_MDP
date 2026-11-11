# Experiment log

Every run records: config, seed, dataset version, split, git commit, library versions, hardware,
runtime, metrics, output path. Nothing here until the first run exists.

| Date | Run | Config | Dataset/split | Result | Notes |
|---|---|---|---|---|---|
| 2026-11-11 | smoke | base.yaml (d64 a1 l1 it10) | ML-1M LLN, 300 test users | static_dot R@10=.0967, static_cos .0600, plan1 .0533 | Paper LLN: Static .0694, Plan-1 .0762. Direction not yet reproduced; leverage ℓ≈1e-2 makes the alignment term inert. |
| 2026-11-11 | sweep | d{32,64,128}×α{1,10}×λ{.01,.1,1} ×η{.1,1,10}, 1500 valid users | ML-1M LLN valid | α=10 lifts static_cos .058→.084; plan1≈static_cos (|Δs|≈0.01 → alignment inert); η=10 hurts at α=1 | Leverage bound ≈ d/n_items for isotropic embeddings explains small-ℓ regime; see DECISIONS.md |
| 2026-11-11 | sweep2 | d{32,64}×α{20,40}×λ{0.1,1}×η{0.1,1}, 1500 valid users | ML-1M LLN valid | best static_cos/plan1: d64 α40 λ1 R=0.0860; static_dot falls to 0.083 at α40 (norms homogenized); no planning gain | α=40 chosen for final runs (best val cosine recall) |
| 2026-11-11 | variants | plan1 post/root/bellman × {d64α10, d128α1}, 1000 valid users | ML-1M LLN valid | all within ±0.002 at d64α10 (post 0.0880, root 0.0900, bellman 0.0890); planning variants HURT at d128α1 (0.042-0.050 vs static_cos 0.061) | 'post' kept as canonical (text-literal); regime-dependent sign of planning effect |
| 2026-11-11 | final test | d64 α40 λ1 it20, full 6035 test users, 3 seeds | ML-1M LLN test | static_dot .0704 (paper .0694 **reproduced**); static_cos .0696; plan1(post) .0694 (paper .0762 NOT reproduced); nDCG ours ~.034 vs paper .0567 | seeds 0/1 nearly identical (strongly convex regime); planning gain absent in our leverage regime |
| 2026-11-11 | plank test | seed 0, 2000-user subset, nsim=100 | ML-1M LLN test | plank R@10=0.0685 nDCG=0.0343 (956ms/user); same-subset: static_dot .0660, static_cos .0685, plan1 .0690 | Plan-K adds nothing over Plan-1 (~680x latency); full set ~100 min/seed at current speed |
