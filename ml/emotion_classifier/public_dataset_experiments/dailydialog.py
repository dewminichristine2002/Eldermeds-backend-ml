"""DailyDialog public-data preparation, contextual input and leakage checks."""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

SOURCE_URL = "https://huggingface.co/datasets/ConvLab/dailydialog"
ARCHIVE_URL = "https://huggingface.co/datasets/ConvLab/dailydialog/resolve/main/data.zip"
LICENSE = "CC BY-NC-SA 4.0"
CITATION = ("Li, Yanran, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. "
            "DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. IJCNLP 2017.")
DATASET_ORIGIN = "public_research"
LABEL_MAP = {
    "no emotion": "neutral",
    "happiness": "happiness",
    "sadness": "sadness",
    "anger": "anger",
}
CLASS_ORDER = ["neutral", "happiness", "sadness", "anger"]
INPUT_FORMATS = ("answer_only", "one_previous", "two_previous")


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_daily_dialog_archive(archive_path: Path):
    archive_path = Path(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if "data/dialogues.json" not in names:
            raise ValueError("DailyDialog archive is missing data/dialogues.json")
        dialogues = json.load(archive.open("data/dialogues.json"))
    if not isinstance(dialogues, list) or not dialogues:
        raise ValueError("DailyDialog dialogues payload must be a nonempty list")
    return dialogues


def prepare_dialogues(dialogues):
    """Preserve original split/dialogue groups and include only defensible labels."""
    records = []
    dialogue_ids = set()
    signatures_by_split = {}
    duplicate_dialogues_dropped = Counter()
    native_label_counts = Counter()
    mapped_label_counts = Counter()
    split_dialogue_counts = Counter()
    split_utterance_counts = Counter()
    omitted_label_counts = Counter()
    source_split_dialogue_counts = Counter(dialogue.get("data_split") for dialogue in dialogues)
    source_native_label_counts = Counter(str(turn.get("emotion") or "").strip().casefold()
                                         for dialogue in dialogues for turn in dialogue.get("turns", []))
    source_split_utterance_counts = Counter({split: sum(len(dialogue.get("turns", [])) for dialogue in dialogues
                                                       if dialogue.get("data_split") == split)
                                             for split in ("train", "validation", "test")})
    split_priority = {"train": 0, "validation": 1, "test": 2}
    ordered_dialogues = sorted(dialogues, key=lambda item: split_priority.get(item.get("data_split"), 9))
    for dialogue in ordered_dialogues:
        split = dialogue.get("data_split")
        dialogue_id = str(dialogue.get("dialogue_id") or "")
        turns = dialogue.get("turns")
        if split not in {"train", "validation", "test"} or not dialogue_id or not isinstance(turns, list) or not turns:
            raise ValueError("invalid DailyDialog dialogue/split structure")
        if dialogue_id in dialogue_ids:
            raise ValueError(f"duplicate dialogue ID across source: {dialogue_id}")
        dialogue_ids.add(dialogue_id)
        signature = "\n".join(str(turn.get("utterance", "")).strip().casefold() for turn in turns)
        signature_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        if signature_hash in signatures_by_split and signatures_by_split[signature_hash] != split:
            duplicate_dialogues_dropped[split] += 1
            continue
        signatures_by_split[signature_hash] = split
        split_dialogue_counts[split] += 1

        for expected_index, turn in enumerate(turns):
            if turn.get("utt_idx") != expected_index:
                raise ValueError(f"dialogue {dialogue_id}: utterance indices must be contiguous and chronological")
            speaker = str(turn.get("speaker") or "").strip()
            utterance = str(turn.get("utterance") or "").strip()
            original_label = str(turn.get("emotion") or "").strip().casefold()
            if speaker not in {"user", "system"} or not utterance or not original_label:
                raise ValueError(f"dialogue {dialogue_id}, turn {expected_index}: missing speaker, text or native label")
            native_label_counts[original_label] += 1
            split_utterance_counts[split] += 1
            mapped_label = LABEL_MAP.get(original_label)
            if mapped_label is None:
                omitted_label_counts[original_label] += 1
                continue
            history = [{"turn_index": prior_index, "speaker": str(turns[prior_index]["speaker"]),
                        "utterance": str(turns[prior_index]["utterance"]).strip()}
                       for prior_index in range(max(0, expected_index - 2), expected_index)]
            sample_key = f"{dialogue_id}:{expected_index}"
            records.append({
                "sample_id": hashlib.sha256(sample_key.encode("utf-8")).hexdigest(),
                "dialogue_id": dialogue_id,
                "native_split": split,
                "turn_index": expected_index,
                "speaker": speaker,
                "current_utterance": utterance,
                "previous_turns": history,
                "original_label": original_label,
                "mapped_label": mapped_label,
                "dataset_origin": DATASET_ORIGIN,
                "source_url": SOURCE_URL,
                "source_license": LICENSE,
                "source_citation": CITATION,
                "population_status": "public_nonclinical_not_eldermeds_data",
            })
            mapped_label_counts[mapped_label] += 1
    return records, {
        "source_dialogues": len(dialogue_ids),
        "source_dialogues_by_split": dict(sorted(source_split_dialogue_counts.items())),
        "source_utterances": sum(source_split_utterance_counts.values()),
        "source_utterances_by_split": dict(sorted(source_split_utterance_counts.items())),
        "retained_dialogues_by_split": dict(sorted(split_dialogue_counts.items())),
        "retained_utterances_by_split": dict(sorted(split_utterance_counts.items())),
        "exact_cross_split_duplicate_dialogues_dropped_by_split": dict(sorted(duplicate_dialogues_dropped.items())),
        "source_native_label_counts": dict(sorted(source_native_label_counts.items())),
        "retained_native_label_counts": dict(sorted(native_label_counts.items())),
        "mapped_records": len(records),
        "mapped_class_counts": {label: mapped_label_counts[label] for label in CLASS_ORDER},
        "excluded_native_label_counts": dict(sorted(omitted_label_counts.items())),
    }


def format_public_input(record, input_format: str, tokenizer=None, max_tokens: int | None = None):
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"unsupported public experiment input format: {input_format}")
    previous = record.get("previous_turns", [])
    if len(previous) > 2:
        raise ValueError("public context is limited to two preceding utterances")
    current_index = record.get("turn_index")
    dialogue_id = record.get("dialogue_id")
    for turn in previous:
        if turn.get("turn_index", current_index) >= current_index:
            raise ValueError("future-turn leakage is forbidden")
        if turn.get("dialogue_id", dialogue_id) != dialogue_id:
            raise ValueError("cross-dialogue context is forbidden")
    selected = [] if input_format == "answer_only" else previous[-1:] if input_format == "one_previous" else previous[-2:]
    current = f"Current speaker: {record['speaker']}\nCurrent utterance: {record['current_utterance']}"

    def render(history):
        lines = [f"Previous turn by {turn['speaker']}: {turn['utterance']}" for turn in history]
        lines.append(current)
        return "\n".join(lines)

    if tokenizer is not None and max_tokens is not None:
        current_tokens = len(tokenizer(current, add_special_tokens=True, truncation=False)["input_ids"])
        if current_tokens > max_tokens:
            raise ValueError(f"sample {record['sample_id']} current utterance exceeds the MiniLM token limit")
        while selected and len(tokenizer(render(selected), add_special_tokens=True, truncation=False)["input_ids"]) > max_tokens:
            selected = selected[1:]
    return render(selected)


def check_dialogue_splits(records):
    groups = {}
    for record in records:
        if record.get("dataset_origin") != DATASET_ORIGIN:
            raise ValueError("public experiment records must identify dataset_origin=public_research")
        key = record["dialogue_id"]
        split = record["native_split"]
        if key in groups and groups[key] != split:
            raise ValueError("a dialogue appears in more than one source split")
        groups[key] = split
        expected_ids = [turn["turn_index"] for turn in record["previous_turns"]]
        if expected_ids != sorted(set(expected_ids)) or any(index >= record["turn_index"] for index in expected_ids):
            raise ValueError("context must be chronological and strictly prior to the current turn")
        if len(expected_ids) > 2:
            raise ValueError("context cannot contain more than two preceding turns")
    return groups
