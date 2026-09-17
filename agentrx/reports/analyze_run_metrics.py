"""
Run Metrics Analysis Script

This script analyzes output directories to compute:
1. Std deviation of correct % accuracy across runs
2. Std deviation for step prediction accuracy (exact, +-1, ..., +-5)
3. Any failure category accuracy (matches any failure for a task_id)
4. Earliest category accuracy (matches first failure for a task_id)
5. Terminal category accuracy (matches last failure for a task_id)
"""

import json
import os
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any

from agentrx.reports.step_accuracy import step_distance_to_nearest_gt

# Failure case number to category mapping
FAILURE_CASE_TO_CATEGORY = {
    1: "Instruction Adherence Failure",
    2: "Invention of New Information",
    3: "Invalid Invocation",
    4: "Misinterpretation of Tool Output",
    5: "Intent Plan Misalignment",
    6: "Underspecified User Intent",
    7: "Intent Not Supported",
    8: "Guardrails Triggered",
    9: "System Failure",
    10: "Inconclusive"
}

# Reverse mapping: category name to number
CATEGORY_TO_FAILURE_CASE = {v: k for k, v in FAILURE_CASE_TO_CATEGORY.items()}

# Ground truth file paths by domain (relative to repository root).
DOMAIN_GROUND_TRUTH_PATHS = {
    "tau":      "data/ground_truth/tau_ground_truth.json",
    "magentic": "data/ground_truth/magentic_one_ground_truth.json",
    "flash":    "data/ground_truth/flash_ground_truth.json",
}

GROUND_TRUTH_PATH = DOMAIN_GROUND_TRUTH_PATHS["tau"]


def normalize_category(category: str) -> str:
    """Normalize category string for matching."""
    if not category:
        return "Unknown"
    
    cat_lower = category.strip().lower()
    
    # Map variations to standard names
    if "instruction" in cat_lower and "adherence" in cat_lower:
        return "Instruction Adherence Failure"
    elif "invention" in cat_lower or ("new" in cat_lower and "information" in cat_lower):
        return "Invention of New Information"
    elif "invalid" in cat_lower and "invocation" in cat_lower:
        return "Invalid Invocation"
    elif "misinterpretation" in cat_lower or "handoff" in cat_lower:
        return "Misinterpretation of Tool Output"
    elif "intent" in cat_lower and ("plan" in cat_lower or "misalignment" in cat_lower):
        return "Intent Plan Misalignment"
    elif "underspecified" in cat_lower or ("user" in cat_lower and "intent" in cat_lower and "not" not in cat_lower):
        return "Underspecified User Intent"
    elif "not supported" in cat_lower or ("intent" in cat_lower and "not" in cat_lower and "supported" in cat_lower):
        return "Intent Not Supported"
    elif "guardrail" in cat_lower:
        return "Guardrails Triggered"
    elif "system" in cat_lower and "failure" in cat_lower:
        return "System Failure"
    elif "inconclusive" in cat_lower:
        return "Inconclusive"
    else:
        return category.strip()


def extract_failure_case_number(failure_case_str) -> int:
    """
    Extract the numeric failure case from string representation.
    
    Args:
        failure_case_str: String like "FailureCase.INVENTION_OF_NEW_INFORMATION",
                         numeric string like "4", or integer like 4
        
    Returns:
        int: Numeric failure case (1-10)
    """
    if isinstance(failure_case_str, int):
        return failure_case_str if 1 <= failure_case_str <= 10 else 10
    
    failure_case_str = str(failure_case_str).strip()
    if failure_case_str.isdigit():
        num = int(failure_case_str)
        return num if 1 <= num <= 10 else 10
    
    # Mapping for enum-style names
    enum_mapping = {
        "INSTRUCTION_ADHERENCE_FAILURE": 1,
        "INSTRUCTION_PLAN_ADHERENCE_FAILURE": 1,
        "INVENTION_OF_NEW_INFORMATION": 2,
        "INVALID_INVOCATION": 3,
        "MISINTERPRETATION_OF_TOOL_OUTPUT": 4,
        "INTENT_PLAN_MISALIGNMENT": 5,
        "UNDERSPECIFIED_USER_INTENT": 6,
        "INTENT_NOT_SUPPORTED": 7,
        "GUARDRAILS_TRIGGERED": 8,
        "SYSTEM_FAILURE": 9,
        "INCONCLUSIVE": 10
    }
    
    if "." in failure_case_str:
        enum_name = failure_case_str.split(".")[-1]
    else:
        enum_name = failure_case_str
    
    upper_str = enum_name.upper().replace(" ", "_").replace("-", "_")
    return enum_mapping.get(upper_str, 10)


def load_ground_truth(gt_path: str) -> Dict[str, Any]:
    """
    Load ground truth data and organize by trajectory_id.
    
    Returns a dict with trajectory_id as key, containing:
    - failures: list of failures sorted by step_number
    - root_cause: root cause failure info
    """
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    gt_by_task = {}
    for entry in gt_data:
        task_id = str(entry['trajectory_id'])
        failures = entry.get('failures', [])
        
        # Sort failures by step_number to get earliest and terminal
        sorted_failures = sorted(failures, key=lambda x: x.get('step_number', 0))
        
        # Get all failure categories for this task
        all_categories = set()
        for f in sorted_failures:
            cat = normalize_category(f.get('failure_category', ''))
            if cat in CATEGORY_TO_FAILURE_CASE:
                all_categories.add(CATEGORY_TO_FAILURE_CASE[cat])
        
        # Get earliest failure category (first by step_number)
        earliest_category = None
        if sorted_failures:
            cat = normalize_category(sorted_failures[0].get('failure_category', ''))
            if cat in CATEGORY_TO_FAILURE_CASE:
                earliest_category = CATEGORY_TO_FAILURE_CASE[cat]
        
        # Get terminal failure category (last by step_number)
        terminal_category = None
        if sorted_failures:
            cat = normalize_category(sorted_failures[-1].get('failure_category', ''))
            if cat in CATEGORY_TO_FAILURE_CASE:
                terminal_category = CATEGORY_TO_FAILURE_CASE[cat]
        
        # Get root cause category and step
        root_cause_id = entry.get('root_cause', {}).get('failure_id')
        root_cause_category = None
        root_cause_step = None
        if root_cause_id:
            for f in sorted_failures:
                if f.get('failure_id') == root_cause_id:
                    cat = normalize_category(f.get('failure_category', ''))
                    if cat in CATEGORY_TO_FAILURE_CASE:
                        root_cause_category = CATEGORY_TO_FAILURE_CASE[cat]
                    root_cause_step = f.get('step_number')
                    break
        
        gt_by_task[task_id] = {
            'failures': sorted_failures,
            'all_categories': all_categories,
            'earliest_category': earliest_category,
            'terminal_category': terminal_category,
            'root_cause_category': root_cause_category,
            'root_cause_step': root_cause_step,
            'root_cause_failure_id': root_cause_id
        }
    
    return gt_by_task


def load_run_results(run_path: str) -> List[Dict]:
    """Load results from a run JSON file."""
    with open(run_path, 'r') as f:
        data = json.load(f)
    return data.get('detailed_results', [])


def load_summary(summary_path: str) -> Dict:
    """Load summary JSON."""
    with open(summary_path, 'r') as f:
        return json.load(f)


def compute_accuracy_std(summary: Dict) -> Tuple[float, float, float]:
    """
    Compute std deviation of correct % accuracy across runs.
    
    Returns: (mean_accuracy, std_accuracy, std_as_percentage)
    """
    run_summaries = summary.get('individual_run_summaries', [])
    
    accuracies = []
    for run in run_summaries:
        correct = run.get('Correct cases', 0)
        incorrect = run.get('Incorrect cases', 0)
        total = correct + incorrect
        if total > 0:
            accuracy = correct / total
            accuracies.append(accuracy)
    
    if len(accuracies) > 0:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=0)  # Population std dev
        std_percentage = std_acc * 100
        return mean_acc, std_acc, std_percentage
    return 0.0, 0.0, 0.0


def compute_avg_step_distance_std(summary: Dict) -> Tuple[float, float, List[float]]:
    """
    Compute mean and std deviation of overall average step distance across runs.
    
    Returns: (mean_distance, std_distance, per_run_distances)
    """
    run_summaries = summary.get('individual_run_summaries', [])
    
    distances = []
    for run in run_summaries:
        dist = run.get('Overall average distance')
        if dist is not None:
            distances.append(dist)
    
    if len(distances) > 0:
        mean_dist = np.mean(distances)
        std_dist = np.std(distances, ddof=0)  # Population std dev
        return mean_dist, std_dist, distances
    return 0.0, 0.0, []


def compute_step_accuracy_std(summary: Dict, output_dir: str = None, gt_by_task: Dict = None, calculate_manually: bool = False) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute std deviation for step prediction accuracies.
    Computes from raw counts (Correct/Incorrect step number predictions) for exact accuracy.
    
    If calculate_manually is True, computes +-1 through +-5 from run files instead of precomputed values.
    
    Returns dict with keys: 'exact', '+-1', '+-2', '+-3', '+-4', '+-5'
    Each value is (mean, std, std_as_percentage)
    """
    run_summaries = summary.get('individual_run_summaries', [])
    
    # For exact step accuracy, compute from raw counts
    exact_accuracies = []
    for run in run_summaries:
        correct = run.get('Correct step number predictions', 0)
        incorrect = run.get('Incorrect step number predictions', 0)
        total = correct + incorrect
        if total > 0:
            exact_accuracies.append(correct / total)
    
    results = {}
    
    # Exact accuracy from raw counts
    if len(exact_accuracies) > 0:
        mean_val = np.mean(exact_accuracies)
        std_val = np.std(exact_accuracies, ddof=0)
        std_percentage = std_val * 100
        results['exact'] = (mean_val, std_val, std_percentage)
    else:
        results['exact'] = (0.0, 0.0, 0.0)
    
    if calculate_manually and output_dir and gt_by_task:
        # Manually calculate +-1 through +-5 from run files
        runs_dir = os.path.join(output_dir, 'runs')
        run_files = ['run1.json', 'run2.json', 'run3.json']
        
        per_run_metrics = {f'+-{i}': [] for i in range(1, 6)}
        
        for run_file in run_files:
            run_path = os.path.join(runs_dir, run_file)
            if not os.path.exists(run_path):
                continue
            
            run_results = load_run_results(run_path)
            
            # Count for each tolerance level
            counts = {f'+-{i}': {'correct': 0, 'total': 0} for i in range(1, 6)}
            
            for result in run_results:
                task_id = str(result.get('task_id', ''))
                if task_id not in gt_by_task:
                    continue
                
                gt_info = gt_by_task[task_id]

                # All labelled GT failure steps for this trajectory. The metric
                # scores the prediction against the NEAREST of these, not the
                # root cause alone — otherwise a correct hit on a non-first GT
                # failure is wrongly counted as a miss.
                gt_steps = [
                    int(f['step_number'])
                    for f in gt_info.get('failures', [])
                    if f.get('step_number') is not None
                ]
                if not gt_steps:
                    continue

                # Get predicted step from result
                predicted_step = result.get('step_mean') or result.get('step_median')
                if predicted_step is None:
                    continue

                # Distance to the nearest GT failure step.
                diff = step_distance_to_nearest_gt(predicted_step, gt_steps)
                
                # Update counts for each tolerance
                for tol in range(1, 6):
                    counts[f'+-{tol}']['total'] += 1
                    if diff <= tol:
                        counts[f'+-{tol}']['correct'] += 1
            
            # Calculate per-run accuracy for each tolerance
            for tol in range(1, 6):
                key = f'+-{tol}'
                if counts[key]['total'] > 0:
                    acc = counts[key]['correct'] / counts[key]['total']
                    per_run_metrics[key].append(acc)
        
        # Compute mean and std for each tolerance
        for key in per_run_metrics:
            values = per_run_metrics[key]
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=0)
                std_percentage = std_val * 100
                results[key] = (mean_val, std_val, std_percentage)
            else:
                results[key] = (0.0, 0.0, 0.0)
    else:
        # Use precomputed values from summary
        metrics = {
            '+-1': [],
            '+-2': [],
            '+-3': [],
            '+-4': [],
            '+-5': []
        }
        
        for run in run_summaries:
            metrics['+-1'].append(run.get('Step accuracy within +-1', 0))
            metrics['+-2'].append(run.get('Step accuracy within +-2', 0))
            metrics['+-3'].append(run.get('Step accuracy within +-3', 0))
            metrics['+-4'].append(run.get('Step accuracy within +-4', 0))
            metrics['+-5'].append(run.get('Step accuracy within +-5', 0))
        
        for key, values in metrics.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=0)
                std_percentage = std_val * 100
                results[key] = (mean_val, std_val, std_percentage)
            else:
                results[key] = (0.0, 0.0, 0.0)
    
    return results


def compute_category_accuracies(output_dir: str, gt_by_task: Dict) -> Dict[str, Dict]:
    """
    Compute the three new category accuracy metrics across all runs.
    
    Returns dict with:
    - 'any_failure': accuracy and details
    - 'earliest': accuracy and details
    - 'terminal': accuracy and details
    """
    runs_dir = os.path.join(output_dir, 'runs')
    
    # Aggregate results across all runs
    all_any_correct = 0
    all_any_total = 0
    all_earliest_correct = 0
    all_earliest_total = 0
    all_terminal_correct = 0
    all_terminal_total = 0
    
    # Per-run metrics for std dev calculation
    per_run_any = []
    per_run_earliest = []
    per_run_terminal = []
    
    run_files = ['run1.json', 'run2.json', 'run3.json']
    
    for run_file in run_files:
        run_path = os.path.join(runs_dir, run_file)
        if not os.path.exists(run_path):
            continue
        
        results = load_run_results(run_path)
        
        run_any_correct = 0
        run_any_total = 0
        run_earliest_correct = 0
        run_earliest_total = 0
        run_terminal_correct = 0
        run_terminal_total = 0
        
        for result in results:
            task_id = str(result.get('task_id', ''))
            if task_id not in gt_by_task:
                continue
            
            gt_info = gt_by_task[task_id]
            
            # Get predicted category from the result
            # The result has 'most_common_failure' or we can look at 'failures' list
            predicted_category = None
            if 'most_common_failure' in result:
                predicted_category = extract_failure_case_number(result['most_common_failure'])
            elif 'failures' in result and len(result['failures']) > 0:
                # Take the first failure's category
                first_failure = result['failures'][0]
                if 'failure_case' in first_failure:
                    predicted_category = extract_failure_case_number(first_failure['failure_case'])
            
            if predicted_category is None:
                continue
            
            # Any Failure Category Accuracy
            if gt_info['all_categories']:
                run_any_total += 1
                if predicted_category in gt_info['all_categories']:
                    run_any_correct += 1
            
            # Earliest Category Accuracy
            if gt_info['earliest_category'] is not None:
                run_earliest_total += 1
                if predicted_category == gt_info['earliest_category']:
                    run_earliest_correct += 1
            
            # Terminal Category Accuracy
            if gt_info['terminal_category'] is not None:
                run_terminal_total += 1
                if predicted_category == gt_info['terminal_category']:
                    run_terminal_correct += 1
        
        # Add to totals
        all_any_correct += run_any_correct
        all_any_total += run_any_total
        all_earliest_correct += run_earliest_correct
        all_earliest_total += run_earliest_total
        all_terminal_correct += run_terminal_correct
        all_terminal_total += run_terminal_total
        
        # Store per-run accuracies
        if run_any_total > 0:
            per_run_any.append(run_any_correct / run_any_total)
        if run_earliest_total > 0:
            per_run_earliest.append(run_earliest_correct / run_earliest_total)
        if run_terminal_total > 0:
            per_run_terminal.append(run_terminal_correct / run_terminal_total)
    
    results = {}
    
    # Any Failure Category
    if all_any_total > 0:
        overall_acc = all_any_correct / all_any_total
        if len(per_run_any) > 0:
            std_val = np.std(per_run_any, ddof=0)
        else:
            std_val = 0.0
        results['any_failure'] = {
            'accuracy': overall_acc,
            'correct': all_any_correct,
            'total': all_any_total,
            'std_dev': std_val,
            'std_dev_percentage': std_val * 100,
            'per_run_accuracies': per_run_any
        }
    else:
        results['any_failure'] = {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'std_dev': 0.0,
            'std_dev_percentage': 0.0,
            'per_run_accuracies': []
        }
    
    # Earliest Category
    if all_earliest_total > 0:
        overall_acc = all_earliest_correct / all_earliest_total
        if len(per_run_earliest) > 0:
            std_val = np.std(per_run_earliest, ddof=0)
        else:
            std_val = 0.0
        results['earliest'] = {
            'accuracy': overall_acc,
            'correct': all_earliest_correct,
            'total': all_earliest_total,
            'std_dev': std_val,
            'std_dev_percentage': std_val * 100,
            'per_run_accuracies': per_run_earliest
        }
    else:
        results['earliest'] = {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'std_dev': 0.0,
            'std_dev_percentage': 0.0,
            'per_run_accuracies': []
        }
    
    # Terminal Category
    if all_terminal_total > 0:
        overall_acc = all_terminal_correct / all_terminal_total
        if len(per_run_terminal) > 0:
            std_val = np.std(per_run_terminal, ddof=0)
        else:
            std_val = 0.0
        results['terminal'] = {
            'accuracy': overall_acc,
            'correct': all_terminal_correct,
            'total': all_terminal_total,
            'std_dev': std_val,
            'std_dev_percentage': std_val * 100,
            'per_run_accuracies': per_run_terminal
        }
    else:
        results['terminal'] = {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'std_dev': 0.0,
            'std_dev_percentage': 0.0,
            'per_run_accuracies': []
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze run metrics from output directory')
    parser.add_argument('output_dir', type=str, 
                        help='Path to output directory (e.g., agent-verify-main/symbolic_invariants/output_results/output_baseline_prompt-gt-False_domain-tau_20260125_142046)')
    parser.add_argument('--domain', type=str, choices=['tau', 'magentic', 'flash'],
                        default='tau', help='Domain to use for ground truth file selection (default: tau)')
    parser.add_argument('--calculate_manually', action='store_true',
                        help='Calculate +-1 through +-5 step accuracies manually from run files instead of using precomputed values')
    
    args = parser.parse_args()
    
    # Determine ground truth file path based on domain
    gt_file_path = DOMAIN_GROUND_TRUTH_PATHS[args.domain]
    
    # Resolve paths
    output_dir = os.path.abspath(args.output_dir)
    summary_path = os.path.join(output_dir, 'analysis', 'summary.json')
    
    if not os.path.exists(summary_path):
        print(f"Error: Summary file not found at {summary_path}")
        return
    
    if not os.path.exists(gt_file_path):
        print(f"Error: Ground truth file not found at {gt_file_path}")
        return
    
    # Load data
    print(f"Loading summary from: {summary_path}")
    summary = load_summary(summary_path)
    
    print(f"Loading ground truth from: {gt_file_path}")
    print(f"Domain: {args.domain}")
    gt_by_task = load_ground_truth(gt_file_path)
    
    print("\n" + "="*80)
    print("RUN METRICS ANALYSIS")
    print("="*80)
    
    # 1. Accuracy std deviation
    print("\n1. ROOT CAUSE ACCURACY (across runs)")
    print("-"*50)
    mean_acc, std_acc, std_pct = compute_accuracy_std(summary)
    print(f"   Mean Accuracy:     {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"   Std Deviation:     {std_acc:.4f}")
    print(f"   Std Dev (%):       {std_pct:.2f}%")
    
    # 1b. Average step distance
    print("\n   AVERAGE STEP DISTANCE (across runs)")
    print("   " + "-"*47)
    mean_dist, std_dist, per_run_dists = compute_avg_step_distance_std(summary)
    print(f"   Mean Avg Distance: {mean_dist:.4f}")
    print(f"   Std Deviation:     {std_dist:.4f}")
    if per_run_dists:
        print(f"   Per-run distances: {[f'{d:.4f}' for d in per_run_dists]}")
    
    # 2. Step prediction accuracy std deviation
    print("\n2. STEP PREDICTION ACCURACY STD DEVIATION")
    if args.calculate_manually:
        print("   (Using manual calculation from run files)")
    print("-"*50)
    step_metrics = compute_step_accuracy_std(summary, output_dir, gt_by_task, args.calculate_manually)
    print(f"   {'Metric':<15} {'Mean':>10} {'Std Dev':>12} {'Std Dev (%)':>12}")
    print(f"   {'-'*49}")
    for key in ['exact', '+-1', '+-2', '+-3', '+-4', '+-5']:
        mean_val, std_val, std_pct = step_metrics[key]
        print(f"   {key:<15} {mean_val:>10.4f} {std_val:>12.4f} {std_pct:>12.2f}%")
    
    # 3. New category accuracy metrics
    print("\n3. CATEGORY ACCURACY METRICS (aggregated across all runs)")
    print("-"*50)
    
    category_metrics = compute_category_accuracies(output_dir, gt_by_task)
    
    # Any Failure Category
    any_m = category_metrics['any_failure']
    print(f"\n   a) ANY FAILURE CATEGORY ACCURACY")
    print(f"      (Correct if prediction matches ANY failure category for task_id)")
    print(f"      Correct: {any_m['correct']} / {any_m['total']}")
    print(f"      Accuracy: {any_m['accuracy']:.4f} ({any_m['accuracy']*100:.2f}%)")
    print(f"      Std Dev across runs: {any_m['std_dev']:.4f} ({any_m['std_dev_percentage']:.2f}%)")
    if any_m['per_run_accuracies']:
        print(f"      Per-run accuracies: {[f'{a:.4f}' for a in any_m['per_run_accuracies']]}")
    
    # Earliest Category
    earliest_m = category_metrics['earliest']
    print(f"\n   b) EARLIEST FAILURE CATEGORY ACCURACY")
    print(f"      (Correct if prediction matches FIRST failure category for task_id)")
    print(f"      Correct: {earliest_m['correct']} / {earliest_m['total']}")
    print(f"      Accuracy: {earliest_m['accuracy']:.4f} ({earliest_m['accuracy']*100:.2f}%)")
    print(f"      Std Dev across runs: {earliest_m['std_dev']:.4f} ({earliest_m['std_dev_percentage']:.2f}%)")
    if earliest_m['per_run_accuracies']:
        print(f"      Per-run accuracies: {[f'{a:.4f}' for a in earliest_m['per_run_accuracies']]}")
    
    # Terminal Category
    terminal_m = category_metrics['terminal']
    print(f"\n   c) TERMINAL FAILURE CATEGORY ACCURACY")
    print(f"      (Correct if prediction matches LAST failure category for task_id)")
    print(f"      Correct: {terminal_m['correct']} / {terminal_m['total']}")
    print(f"      Accuracy: {terminal_m['accuracy']:.4f} ({terminal_m['accuracy']*100:.2f}%)")
    print(f"      Std Dev across runs: {terminal_m['std_dev']:.4f} ({terminal_m['std_dev_percentage']:.2f}%)")
    if terminal_m['per_run_accuracies']:
        print(f"      Per-run accuracies: {[f'{a:.4f}' for a in terminal_m['per_run_accuracies']]}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()