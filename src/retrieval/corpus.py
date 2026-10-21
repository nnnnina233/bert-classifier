import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_labels(path):
    labels = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        number, label = line.split("\t", 1)
        index = int(number)
        if index in labels or not label.strip():
            raise ValueError("标签映射存在重复编号或空标签。")
        labels[index] = label.strip()
    if not labels or sorted(labels) != list(range(len(labels))) or len(set(labels.values())) != len(labels):
        raise ValueError("标签编号必须从 0 连续排列，类别名称不能重复。")
    return [labels[i] for i in range(len(labels))]


def prepare_corpus(source, labels, max_per_class=200, seed=42):
    if max_per_class < 0:
        raise ValueError("max_per_class 不能小于 0。")
    label_ids = {label: i for i, label in enumerate(labels)}
    titles = defaultdict(set)
    stats = {"rows": 0, "empty": 0, "duplicates": 0, "conflicting_titles": 0}
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"label", "text_a"}.issubset(reader.fieldnames or []):
            raise ValueError("训练集必须包含 label 和 text_a 两列。")
        for row in reader:
            stats["rows"] += 1
            text, label = (row.get("text_a") or "").strip(), (row.get("label") or "").strip()
            if not text:
                stats["empty"] += 1
                continue
            if label not in label_ids:
                raise ValueError(f"训练集标签不在映射中：{label}")
            if label in titles[text]:
                stats["duplicates"] += 1
            titles[text].add(label)
    grouped = defaultdict(list)
    for text, matches in sorted(titles.items()):
        if len(matches) != 1:
            stats["conflicting_titles"] += 1
            continue
        label = next(iter(matches))
        source_id = hashlib.sha256(json.dumps([text, label], ensure_ascii=False).encode()).hexdigest()
        grouped[label].append({"source_id": source_id, "text": text, "label": label, "label_id": label_ids[label]})
    rng = random.Random(seed)
    records = []
    for label in labels:
        candidates = grouped[label]
        if max_per_class and len(candidates) > max_per_class:
            candidates = rng.sample(candidates, max_per_class)
        records.extend(sorted(candidates, key=lambda item: item["source_id"]))
    if not records:
        raise ValueError("清理后没有可用于索引的商品。")
    stats["indexed"] = len(records)
    stats["per_class"] = {label: sum(item["label"] == label for item in records) for label in labels}
    return records, stats
