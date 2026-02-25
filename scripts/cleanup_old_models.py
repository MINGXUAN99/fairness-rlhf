#!/usr/bin/env python3
"""
Cleanup script to remove old models and datasets from wandb and Hugging Face Hub.

This script helps you:
1. List all models in your Hugging Face Hub account
2. List all datasets in your Hugging Face Hub account
3. List all wandb runs
4. Delete models/datasets/runs (interactive or dry-run)

Usage:
    python scripts/cleanup_old_models.py --user_id YOUR_USER_ID [options]

Examples:
    # List and interactively delete HF models + datasets
    python scripts/cleanup_old_models.py --user_id MY_USER --hf_datasets

    # Only list/delete HF datasets (no models, no wandb)
    python scripts/cleanup_old_models.py --user_id MY_USER --hf_datasets_only

    # List only (no deletion)
    python scripts/cleanup_old_models.py --user_id MY_USER --hf_datasets --list_only
"""

import sys
import os
import argparse
from datetime import datetime
from typing import List, Optional

# Add project root to path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))
sys.path.insert(0, ROOT)

try:
    from huggingface_hub import HfApi
    import wandb
except ImportError as e:
    print(f"Error: Missing required package. Please install: {e}")
    print("Run: pip install huggingface-hub wandb")
    sys.exit(1)


def format_size(size_bytes: Optional[int]) -> str:
    """Format bytes to human-readable size."""
    if size_bytes is None:
        return "Unknown"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def _safe_get_wandb_default_entity(api: "wandb.Api") -> Optional[str]:
    """
    Best-effort default entity resolution for wandb.
    Prefer explicit env vars, then Api settings/default_entity.
    """
    return (
        os.getenv("WANDB_ENTITY")
        or getattr(api, "settings", {}).get("entity")  # type: ignore[union-attr]
        or getattr(api, "default_entity", None)
    )


def _hf_repo_total_size_bytes(api: HfApi, repo_id: str) -> Optional[int]:
    """
    Compute total repo size for a HF model repo.
    Uses repo_info(files_metadata=True) so `siblings[*].size` is populated.
    """
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    except Exception:
        # Fallback to model_info without metadata (often won't include sizes)
        try:
            info = api.model_info(repo_id)
        except Exception:
            return None

    total = 0
    siblings = getattr(info, "siblings", None)
    if not siblings:
        return None
    for sib in siblings:
        size = getattr(sib, "size", None)
        if isinstance(size, int) and size > 0:
            total += size
    return total if total > 0 else None


def list_hf_models(user_id: str, pattern: Optional[str] = None) -> List[dict]:
    """List all models in Hugging Face Hub for the given user."""
    print(f"\n{'='*80}")
    print(f"Listing Hugging Face models for user: {user_id}")
    print(f"{'='*80}\n")
    
    api = HfApi()
    models = []
    
    try:
        # List all models for the user
        # Use authenticated API to include private repos if you are logged in (`huggingface-cli login`).
        all_models = api.list_models(author=user_id, sort="last_modified", token=True)
        
        for model in all_models:
            model_id = model.id
            # Filter by pattern if provided
            if pattern and pattern not in model_id:
                continue
            
            try:
                model_info = api.model_info(model_id)
                last_modified = getattr(model_info, "last_modified", None)
                total_size = _hf_repo_total_size_bytes(api, model_id)
                
                models.append({
                    'id': model_id,
                    'last_modified': last_modified,
                    'private': model.private if hasattr(model, 'private') else False,
                    'size': total_size,
                })
            except Exception as e:
                print(f"Warning: Could not get info for {model_id}: {e}")
                models.append({
                    'id': model_id,
                    'last_modified': None,
                    'private': None,
                    'size': None,
                })
        
        print(f"Found {len(models)} model(s):\n")
        for i, model in enumerate(models, 1):
            modified_str = model['last_modified'].strftime("%Y-%m-%d %H:%M:%S") if model['last_modified'] else "Unknown"
            private_str = "private" if model['private'] else "public"
            size_str = format_size(model['size'])
            print(f"  {i:3d}. {model['id']}")
            print(f"       Last modified: {modified_str} | {private_str} | Size: {size_str}")
        
    except Exception as e:
        print(f"Error listing models: {e}")
        return []
    
    return models


def list_hf_datasets(user_id: str, pattern: Optional[str] = None) -> List[dict]:
    """List all datasets in Hugging Face Hub for the given user."""
    print(f"\n{'='*80}")
    print(f"Listing Hugging Face datasets for user: {user_id}")
    print(f"{'='*80}\n")

    api = HfApi()
    datasets_out: List[dict] = []

    try:
        all_datasets = api.list_datasets(author=user_id, sort="last_modified", token=True)

        for ds in all_datasets:
            ds_id = ds.id
            if pattern and pattern not in ds_id:
                continue

            try:
                info = api.repo_info(repo_id=ds_id, repo_type="dataset", files_metadata=True)
                last_modified = getattr(info, "last_modified", None)
                total_size = 0
                siblings = getattr(info, "siblings", None)
                if siblings:
                    for sib in siblings:
                        size = getattr(sib, "size", None)
                        if isinstance(size, int) and size > 0:
                            total_size += size
                datasets_out.append(
                    {
                        "id": ds_id,
                        "last_modified": last_modified,
                        "private": getattr(ds, "private", False),
                        "size": total_size if total_size > 0 else None,
                    }
                )
            except Exception as e:
                print(f"Warning: Could not get info for dataset {ds_id}: {e}")
                datasets_out.append(
                    {
                        "id": ds_id,
                        "last_modified": None,
                        "private": None,
                        "size": None,
                    }
                )

        print(f"Found {len(datasets_out)} dataset(s):\n")
        for i, ds in enumerate(datasets_out, 1):
            modified_str = ds["last_modified"].strftime("%Y-%m-%d %H:%M:%S") if ds["last_modified"] else "Unknown"
            private_str = "private" if ds["private"] else "public"
            size_str = format_size(ds["size"])
            print(f"  {i:3d}. {ds['id']}")
            print(f"       Last modified: {modified_str} | {private_str} | Size: {size_str}")

    except Exception as e:
        print(f"Error listing datasets: {e}")
        return []

    return datasets_out


def delete_hf_datasets(user_id: str, dataset_ids: List[str], datasets: List[dict], dry_run: bool = True) -> None:
    """Delete datasets from Hugging Face Hub."""
    api = HfApi()

    print(f"\n{'='*80}")
    if dry_run:
        print(f"DRY RUN: Would delete {len(dataset_ids)} dataset(s) from Hugging Face")
    else:
        print(f"Deleting {len(dataset_ids)} dataset(s) from Hugging Face")
    print(f"{'='*80}\n")

    total_size = 0
    for ds_id in dataset_ids:
        ds = next((d for d in datasets if d["id"] == ds_id), None)
        size_str = format_size(ds["size"]) if ds and ds.get("size") else "Unknown"
        if ds and ds.get("size"):
            total_size += ds["size"]

        if dry_run:
            print(f"  [DRY RUN] Would delete: {ds_id} (Size: {size_str})")
        else:
            try:
                api.delete_repo(repo_id=ds_id, repo_type="dataset")
                print(f"  ✓ Deleted: {ds_id} (Size: {size_str})")
            except Exception as e:
                print(f"  ✗ Failed to delete {ds_id}: {e}")

    if total_size > 0:
        print(f"\nTotal size freed: {format_size(total_size)}")


def interactive_delete_hf_datasets(datasets: List[dict], user_id: str) -> None:
    """Interactive deletion of Hugging Face datasets."""
    if not datasets:
        print("No datasets to delete.")
        return

    print("\n" + "=" * 80)
    print("INTERACTIVE DELETION - Hugging Face Datasets")
    print("=" * 80)
    print("\nEnter dataset numbers to delete (comma-separated, e.g., 1,3,5)")
    print("Or enter 'all' to delete all, 'pattern:<pattern>' to filter by pattern")
    print("Or press Enter to skip")

    response = input("\nSelection: ").strip()
    if not response:
        print("Skipping deletion.")
        return

    if response.lower() == "all":
        dataset_ids = [d["id"] for d in datasets]
    elif response.startswith("pattern:"):
        patt = response.split(":", 1)[1]
        dataset_ids = [d["id"] for d in datasets if patt in d["id"]]
        print(f"Filtered to {len(dataset_ids)} datasets matching pattern: {patt}")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in response.split(",")]
            dataset_ids = [datasets[i]["id"] for i in indices if 0 <= i < len(datasets)]
        except ValueError:
            print("Invalid input. Skipping deletion.")
            return

    if not dataset_ids:
        print("No datasets selected.")
        return

    print(f"\nSelected {len(dataset_ids)} dataset(s) for deletion:")
    total_size = 0
    for ds_id in dataset_ids:
        ds = next((d for d in datasets if d["id"] == ds_id), None)
        size_str = format_size(ds["size"]) if ds and ds.get("size") else "Unknown"
        if ds and ds.get("size"):
            total_size += ds["size"]
        print(f"  - {ds_id} (Size: {size_str})")
    if total_size > 0:
        print(f"\nTotal size to be freed: {format_size(total_size)}")

    confirm = input("\nConfirm deletion? (yes/no): ").strip().lower()
    if confirm == "yes":
        delete_hf_datasets(user_id, dataset_ids, datasets, dry_run=False)
    else:
        print("Deletion cancelled.")


def delete_hf_models(user_id: str, model_ids: List[str], models: List[dict], dry_run: bool = True) -> None:
    """Delete models from Hugging Face Hub."""
    api = HfApi()
    
    print(f"\n{'='*80}")
    if dry_run:
        print(f"DRY RUN: Would delete {len(model_ids)} model(s) from Hugging Face")
    else:
        print(f"Deleting {len(model_ids)} model(s) from Hugging Face")
    print(f"{'='*80}\n")
    
    total_size = 0
    for model_id in model_ids:
        model = next((m for m in models if m['id'] == model_id), None)
        size_str = format_size(model['size']) if model and model.get('size') else "Unknown"
        if model and model.get('size'):
            total_size += model['size']
        
        if dry_run:
            print(f"  [DRY RUN] Would delete: {model_id} (Size: {size_str})")
        else:
            try:
                api.delete_repo(repo_id=model_id, repo_type="model")
                print(f"  ✓ Deleted: {model_id} (Size: {size_str})")
            except Exception as e:
                print(f"  ✗ Failed to delete {model_id}: {e}")
    
    if total_size > 0:
        print(f"\nTotal size freed: {format_size(total_size)}")


def list_wandb_runs(entity: Optional[str] = None, project: Optional[str] = None, 
                    pattern: Optional[str] = None) -> List[dict]:
    """List all wandb runs."""
    print(f"\n{'='*80}")
    print("Listing wandb runs")
    if entity:
        print(f"Entity: {entity}")
    if project:
        print(f"Project: {project}")
    print(f"{'='*80}\n")
    
    try:
        # Initialize wandb API
        api = wandb.Api()

        # Resolve entity/project defaults from env / wandb settings
        entity = entity or _safe_get_wandb_default_entity(api)
        project = project or os.getenv("WANDB_PROJECT")

        if not entity:
            print("Error: Could not determine wandb entity.")
            print("Set WANDB_ENTITY or pass --wandb_entity (and optionally --wandb_project).")
            return []

        print(f"Resolved wandb entity: {entity}")
        if project:
            print(f"Resolved wandb project: {project}")
        print("")
        
        # Get runs
        runs = []
        if entity and project:
            try:
                runs = api.runs(f"{entity}/{project}")
            except Exception as e:
                print(f"Warning: Could not access {entity}/{project}: {e}")
        elif entity:
            # List all projects for entity, then get runs
            try:
                projects_list = list(api.projects(entity))
                print(f"Found {len(projects_list)} project(s) for entity '{entity}'\n")
                for proj in projects_list:
                    try:
                        proj_runs = api.runs(f"{entity}/{proj.name}")
                        runs.extend(proj_runs)
                    except Exception as e:
                        print(f"Warning: Could not access runs in project {proj.name}: {e}")
            except Exception as e:
                print(f"Warning: Could not list projects for entity '{entity}': {e}")
        
        def _parse_created_at(val) -> Optional[datetime]:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                # wandb sometimes returns ISO strings like "2026-02-14T00:00:00"
                try:
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                except Exception:
                    return None
            return None

        run_list = []
        for run in runs:
            run_name = run.name
            run_id = run.id
            
            # Filter by pattern if provided
            if pattern and pattern not in run_name:
                continue
            
            created_at_raw = getattr(run, "created_at", None)
            created_at_dt = _parse_created_at(created_at_raw)
            state = run.state if hasattr(run, 'state') else "unknown"
            
            # Calculate total size from run files
            total_size = 0
            try:
                # Get files for this run
                files = run.files()
                for file in files:
                    if hasattr(file, 'size') and file.size:
                        total_size += file.size
            except Exception as e:
                # Some runs might not have accessible files
                pass
            
            run_list.append({
                'id': run_id,
                'name': run_name,
                'created_at': created_at_raw,
                '_created_at_dt': created_at_dt,
                'state': state,
                'entity': getattr(run, "entity", None) or entity,
                'project': getattr(run, "project", None),
                'size': total_size if total_size > 0 else None,
            })
        
        # Sort by creation date (newest first)
        run_list.sort(key=lambda x: x.get('_created_at_dt') or datetime.min, reverse=True)
        
        print(f"Found {len(run_list)} run(s):\n")
        for i, run in enumerate(run_list, 1):
            if run.get('_created_at_dt'):
                created_str = run['_created_at_dt'].strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(run.get('created_at'), str):
                created_str = run['created_at']
            else:
                created_str = "Unknown"
            size_str = format_size(run['size'])
            print(f"  {i:3d}. {run['name']}")
            print(f"       ID: {run['id']} | Created: {created_str} | State: {run['state']} | Size: {size_str}")
            if run['entity'] and run['project']:
                print(f"       Entity/Project: {run['entity']}/{run['project']}")
        
    except Exception as e:
        print(f"Error listing wandb runs: {e}")
        print("Make sure you're logged in to wandb: wandb login")
        print("You may also need to specify --wandb_entity and --wandb_project")
        return []
    
    return run_list


def delete_wandb_runs(runs: List[dict], run_ids: List[str], dry_run: bool = True) -> None:
    """Delete wandb runs."""
    api = wandb.Api()
    
    print(f"\n{'='*80}")
    if dry_run:
        print(f"DRY RUN: Would delete {len(run_ids)} wandb run(s)")
    else:
        print(f"Deleting {len(run_ids)} wandb run(s)")
    print(f"{'='*80}\n")
    
    total_size = 0
    for run_id in run_ids:
        # Find the run info to get entity/project
        run_info = next((r for r in runs if r['id'] == run_id), None)
        if not run_info:
            print(f"  ✗ Run {run_id} not found in list")
            continue
        
        size_str = format_size(run_info.get('size')) if run_info.get('size') else "Unknown"
        if run_info.get('size'):
            total_size += run_info['size']
        
        if dry_run:
            print(f"  [DRY RUN] Would delete run: {run_info['name']} (ID: {run_id}, Size: {size_str})")
        else:
            try:
                # Construct the full path: entity/project/run_id
                if run_info['entity'] and run_info['project']:
                    run_path = f"{run_info['entity']}/{run_info['project']}/{run_id}"
                else:
                    run_path = run_id
                
                run = api.run(run_path)
                run.delete()
                print(f"  ✓ Deleted run: {run_info['name']} (ID: {run_id}, Size: {size_str})")
            except Exception as e:
                print(f"  ✗ Failed to delete run {run_id}: {e}")
    
    if total_size > 0:
        print(f"\nTotal size freed: {format_size(total_size)}")


def interactive_delete_hf(models: List[dict], user_id: str) -> None:
    """Interactive deletion of Hugging Face models."""
    if not models:
        print("No models to delete.")
        return
    
    print("\n" + "="*80)
    print("INTERACTIVE DELETION - Hugging Face Models")
    print("="*80)
    print("\nEnter model numbers to delete (comma-separated, e.g., 1,3,5)")
    print("Or enter 'all' to delete all, 'pattern:<pattern>' to filter by pattern")
    print("Or press Enter to skip")
    
    response = input("\nSelection: ").strip()
    
    if not response:
        print("Skipping deletion.")
        return
    
    if response.lower() == 'all':
        model_ids = [m['id'] for m in models]
    elif response.startswith('pattern:'):
        pattern = response.split(':', 1)[1]
        model_ids = [m['id'] for m in models if pattern in m['id']]
        print(f"Filtered to {len(model_ids)} models matching pattern: {pattern}")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in response.split(',')]
            model_ids = [models[i]['id'] for i in indices if 0 <= i < len(models)]
        except ValueError:
            print("Invalid input. Skipping deletion.")
            return
    
    if not model_ids:
        print("No models selected.")
        return
    
    print(f"\nSelected {len(model_ids)} model(s) for deletion:")
    total_size = 0
    for model_id in model_ids:
        model = next((m for m in models if m['id'] == model_id), None)
        size_str = format_size(model['size']) if model and model.get('size') else "Unknown"
        if model and model.get('size'):
            total_size += model['size']
        print(f"  - {model_id} (Size: {size_str})")
    if total_size > 0:
        print(f"\nTotal size to be freed: {format_size(total_size)}")
    
    confirm = input("\nConfirm deletion? (yes/no): ").strip().lower()
    if confirm == 'yes':
        delete_hf_models(user_id, model_ids, models, dry_run=False)
    else:
        print("Deletion cancelled.")


def interactive_delete_wandb(runs: List[dict]) -> None:
    """Interactive deletion of wandb runs."""
    if not runs:
        print("No runs to delete.")
        return
    
    print("\n" + "="*80)
    print("INTERACTIVE DELETION - wandb Runs")
    print("="*80)
    print("\nEnter run numbers to delete (comma-separated, e.g., 1,3,5)")
    print("Or enter 'all' to delete all, 'pattern:<pattern>' to filter by pattern")
    print("Or press Enter to skip")
    
    response = input("\nSelection: ").strip()
    
    if not response:
        print("Skipping deletion.")
        return
    
    if response.lower() == 'all':
        run_ids = [r['id'] for r in runs]
    elif response.startswith('pattern:'):
        pattern = response.split(':', 1)[1]
        run_ids = [r['id'] for r in runs if pattern in r['name']]
        print(f"Filtered to {len(run_ids)} runs matching pattern: {pattern}")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in response.split(',')]
            run_ids = [runs[i]['id'] for i in indices if 0 <= i < len(runs)]
        except ValueError:
            print("Invalid input. Skipping deletion.")
            return
    
    if not run_ids:
        print("No runs selected.")
        return
    
    print(f"\nSelected {len(run_ids)} run(s) for deletion:")
    total_size = 0
    for run_id in run_ids:
        run = next(r for r in runs if r['id'] == run_id)
        size_str = format_size(run.get('size')) if run.get('size') else "Unknown"
        if run.get('size'):
            total_size += run['size']
        print(f"  - {run['name']} (ID: {run_id}, Size: {size_str})")
    if total_size > 0:
        print(f"\nTotal size to be freed: {format_size(total_size)}")
    
    confirm = input("\nConfirm deletion? (yes/no): ").strip().lower()
    if confirm == 'yes':
        delete_wandb_runs(runs, run_ids, dry_run=False)
    else:
        print("Deletion cancelled.")


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup old models from wandb and Hugging Face Hub"
    )
    parser.add_argument("--user_id", type=str, required=True,
                        help="Hugging Face user ID")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity (default: from wandb config)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project (default: from wandb config)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Filter models/runs by pattern (substring match)")
    parser.add_argument("--hf_only", action="store_true",
                        help="Only list/delete Hugging Face models (no wandb)")
    parser.add_argument("--hf_datasets", action="store_true",
                        help="Include Hugging Face datasets (list/delete) alongside models")
    parser.add_argument("--hf_datasets_only", action="store_true",
                        help="Only list/delete Hugging Face datasets (no models, no wandb)")
    parser.add_argument("--wandb_only", action="store_true",
                        help="Only list/delete wandb runs")
    parser.add_argument("--list_only", action="store_true",
                        help="Only list, don't delete")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()

    # Only Hugging Face datasets (no models, no wandb)
    if args.hf_datasets_only:
        hf_datasets = list_hf_datasets(args.user_id, pattern=args.pattern)
        if not args.list_only and hf_datasets:
            interactive_delete_hf_datasets(hf_datasets, args.user_id)
        print("\n" + "="*80)
        print("Cleanup complete!")
        print("="*80)
        return

    # List Hugging Face models
    if not args.wandb_only:
        hf_models = list_hf_models(args.user_id, pattern=args.pattern)
        
        if not args.list_only and hf_models:
            interactive_delete_hf(hf_models, args.user_id)

        if args.hf_datasets:
            hf_datasets = list_hf_datasets(args.user_id, pattern=args.pattern)
            if not args.list_only and hf_datasets:
                interactive_delete_hf_datasets(hf_datasets, args.user_id)
    
    # List wandb runs
    if not args.hf_only:
        wandb_runs = list_wandb_runs(
            entity=args.wandb_entity,
            project=args.wandb_project,
            pattern=args.pattern
        )
        
        if not args.list_only and wandb_runs:
            interactive_delete_wandb(wandb_runs)
    
    print("\n" + "="*80)
    print("Cleanup complete!")
    print("="*80)


if __name__ == "__main__":
    main()

