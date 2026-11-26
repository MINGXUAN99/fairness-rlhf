import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import argparse
import pandas as pd
from datasets import load_dataset
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Read and analyze evaluation scores from scored datasets.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--proportion", type=float, default=None, help="Proportion of minority group (required for all methods).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for scores (default: 1.0).")
    parser.add_argument("--method", type=str, required=True, choices=["vanilla", "maxmin", "pcgrad", "sumgrad", "mgda"])
    parser.add_argument("--output_csv", type=str, default=None, help="Optional: Save results to CSV file.")
    return parser.parse_args()


def load_scored_dataset(user_id, method, seed, proportion, alpha):
    """Load the scored evaluation dataset from HuggingFace Hub."""
    if method not in ["vanilla", "maxmin", "pcgrad", "sumgrad", "mgda"]:
        raise ValueError(f"Unknown method: {method}")
    dataset_name = f"{user_id}/{method}-imdb-ppo-eval-prop{proportion}-alpha{alpha}-seed{seed}"

    # Load dataset - handle both DatasetDict and Dataset cases
    loaded_dataset = load_dataset(dataset_name)
    # If it's a DatasetDict, select the first available split (usually 'train')
    if hasattr(loaded_dataset, 'keys') and len(loaded_dataset.keys()) > 0:
        split_name = list(loaded_dataset.keys())[0]
        dataset = loaded_dataset[split_name]
    else:
        dataset = loaded_dataset
    
    return dataset


def analyze_scores(dataset):
    """Analyze the evaluation scores in the dataset."""
    print("=" * 80)
    print("EVALUATION SCORES ANALYSIS")
    print("=" * 80)
    
    # Convert to pandas for easier analysis
    df = dataset.to_pandas()
    
    print(f"\nDataset size: {len(df)} examples")
    print(f"\nColumns: {list(df.columns)}")
    
    # Check available metrics
    metrics = {}
    if "positiveness" in df.columns:
        metrics["positiveness"] = df["positiveness"]
    if "length" in df.columns:
        metrics["length"] = df["length"]
    if "normalized_length" in df.columns:
        metrics["normalized_length"] = df["normalized_length"]
    
    print("\n" + "=" * 80)
    print("METRIC STATISTICS")
    print("=" * 80)
    
    for metric_name, values in metrics.items():
        # Convert to numpy array and flatten if needed (handle lists/arrays)
        values_array = np.array(values.tolist() if hasattr(values, 'tolist') else values)
        if values_array.ndim > 1:
            values_array = values_array.flatten()
        # Filter out any NaN or None values
        values_array = values_array[~np.isnan(values_array)]
        
        if len(values_array) == 0:
            print(f"\n{metric_name.upper()}: No valid values found")
            continue
            
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean:   {float(np.mean(values_array)):.4f}")
        print(f"  Std:    {float(np.std(values_array)):.4f}")
        print(f"  Min:    {float(np.min(values_array)):.4f}")
        print(f"  Max:    {float(np.max(values_array)):.4f}")
        print(f"  Median: {float(np.median(values_array)):.4f}")
        print(f"  25th percentile: {float(np.percentile(values_array, 25)):.4f}")
        print(f"  75th percentile: {float(np.percentile(values_array, 75)):.4f}")
    
    # Show sample data
    print("\n" + "=" * 80)
    print("SAMPLE DATA (first 5 examples)")
    print("=" * 80)
    
    # Select relevant columns for display
    display_cols = ["prompt"] if "prompt" in df.columns else []
    display_cols.extend(["completion"] if "completion" in df.columns else [])
    display_cols.extend(list(metrics.keys()))
    
    if display_cols:
        print(df[display_cols].head().to_string())
    
    # Correlation analysis if multiple metrics exist
    if len(metrics) > 1:
        print("\n" + "=" * 80)
        print("METRIC CORRELATIONS")
        print("=" * 80)
        # Convert metrics to proper format for correlation
        metric_dict = {}
        min_length = len(df)
        for metric_name, values in metrics.items():
            values_array = np.array(values.tolist() if hasattr(values, 'tolist') else values)
            if values_array.ndim > 1:
                values_array = values_array.flatten()
            # Take only the valid length (should match dataframe length)
            values_array = values_array[:len(df)]
            metric_dict[metric_name] = values_array
            min_length = min(min_length, len(values_array))
        
        # Ensure all metrics have the same length
        for metric_name in metric_dict:
            metric_dict[metric_name] = metric_dict[metric_name][:min_length]
        
        metric_df = pd.DataFrame(metric_dict)
        print(metric_df.corr().to_string())
    
    return df


if __name__ == "__main__":
    args = parse_args()
    
    # Validate required arguments
    if args.method in ["vanilla", "maxmin", "pcgrad", "sumgrad", "mgda"] and args.proportion is None:
        raise ValueError(f"--proportion is required for {args.method} method")
    
    # Load dataset
    print(f"Loading evaluation dataset for method: {args.method}")
    dataset = load_scored_dataset(
        args.user_id, args.method, args.seed, 
        args.proportion, args.alpha
    )
    
    # Analyze scores
    df = analyze_scores(dataset)
    
    # Save to CSV if requested
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nDataset saved to: {args.output_csv}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
