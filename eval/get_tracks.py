import csv

fmap = "F:/ResearchProject/eval/annotation_package/frame_track_map.csv"
ids = set()
with open(fmap) as f:
    for row in csv.DictReader(f):
        ids.add(int(row["track_id"]))

print("Unique track IDs in interaction video1:")
print(sorted(ids))
print(f"Total: {len(ids)} tracks")
