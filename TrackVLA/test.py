import os, gzip, json

scenes_dir = "/data1/szz2026/TrackVLA/data/scene_datasets"
json_gz    = "/data1/szz2026/TrackVLA/data/datasets/track/AT/train/train.json.gz"

with gzip.open(json_gz, "rt", encoding="utf-8") as f:
    data = json.load(f)

eps = data["episodes"] if isinstance(data, dict) and "episodes" in data else data
print("episodes_total =", len(eps))

missing = []
for ep in eps[:200]:  # 先抽前200条
    sid = ep["scene_id"]
    p = sid if os.path.isabs(sid) else os.path.join(scenes_dir, sid)
    if not os.path.exists(p):
        missing.append(p)

print("missing_in_first_200 =", len(missing))
print("sample_missing_paths:")
for p in missing[:5]:
    print("  ", p)