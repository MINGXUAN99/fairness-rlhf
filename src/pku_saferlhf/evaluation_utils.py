"""File contracts and checks for reproducible policy comparisons."""
import hashlib
import json
from pathlib import Path


def prompt_id(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def resolve_revision(model, revision="main", repo_type="model"):
    if Path(model).is_dir():
        return None
    from huggingface_hub import HfApi
    return HfApi().repo_info(model, repo_type=repo_type, revision=revision).sha


def validate_records(records, manifest):
    if manifest.get("status") != "complete":
        raise ValueError("Input run is incomplete")
    expected_runs = {run["run_id"] for run in manifest["runs"]}
    expected_prompts = set(manifest["prompt_ids"])
    if len(expected_runs) != len(manifest["runs"]) or len(expected_prompts) != len(manifest["prompt_ids"]):
        raise ValueError("Manifest contains duplicate run or prompt IDs")
    if not expected_runs or not expected_prompts:
        raise ValueError("No runs or prompts")
    keys, texts = set(), {}
    for row in records:
        key = (row["run_id"], row["prompt_id"])
        if key in keys:
            raise ValueError(f"Duplicate response: {key}")
        keys.add(key)
        if row["prompt_id"] != prompt_id(row["prompt"]):
            raise ValueError("Prompt hash does not match text")
        if not isinstance(row["response"], str):
            raise ValueError("Response must be text")
        texts[row["prompt_id"]] = row["prompt"]
    if keys != {(run, prompt) for run in expected_runs for prompt in expected_prompts}:
        raise ValueError("Runs must contain exactly the same manifest prompts")


def load_records(directory, filename):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    records = [json.loads(line) for line in (directory / filename).read_text().splitlines() if line.strip()]
    validate_records(records, manifest)
    return records, manifest
