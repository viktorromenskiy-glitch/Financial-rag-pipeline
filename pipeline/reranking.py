import json
import datetime

run_config = {
    "step": "step3_hybrid_retrieval",
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "pool_size": 50,
    "vector_weight": 0.5,
    "text_weight": 0.5,
    "num_candidates_multiplier": 20,
    "eval_set_size": 250,
    "recall_at_5": 0.792,
    "hits": 198,
    "note": (
        "Historical checkpoint 0.808 (202/250) was measured in "
        "experiments_weeks_1_2.ipynb with pool_size=10 (numCandidates=50, "
        "limit=10 per branch before $rankFusion) - exactly reproduced today "
        "with those parameters (202/250). 0.792 at pool_size=50 is the "
        "correct baseline for this codebase going into Step 4, since "
        "pool=50 is what the reranker consumes."
    ),
}

with open('/content/drive/MyDrive/RAG-project/data/t2-ragbench/run_config_step3.json', 'w') as f:
    json.dump(run_config, f, indent=2)

print(json.dumps(run_config, indent=2))
