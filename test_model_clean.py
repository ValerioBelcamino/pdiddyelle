# Import necessary libraries
import argparse
import os
import re
import json
import time
import pickle
from collections import defaultdict
from pathlib import Path

import requests
import partial_plan_progressor as plan_progressor

# Import necessary functions
from utils import (
    Folders_Structure,
    open_json_file,
    write_domain_and_create_logs_dir,
    write_problem,
    write_plan,
    calculate_statistics,
    final_output,
    write_log
)
 
# Project metadata
__author__ = "Nicholas Attolino"
__copyright__ = "Copyright 2024, Nicholas Attolino"
__license__ = "GNU"
__version__ = "1.0.0"
__maintainer__ = "Nicholas Attolino"
__email__ = "nicholasattolino@gmail.com"
__status__ = "Development"


plan_progressor.exit = lambda *args, **kwargs: None
validate_plan_text = plan_progressor.validate_plan_text

MAX_RETRIES = 5
VAL_TIMEOUT_DEFAULT = 10
PLAN_STEP_INCREMENT = 0.002


def sanitize_plan_lines(plan_text):
    lines = []
    for raw_line in plan_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.upper() == "END":
            continue
        lines.append(stripped)
    return lines


def build_plan_text(plan_lines):
    if not plan_lines:
        return ""
    return "\n".join(plan_lines) + "\nEND"


def shift_plan_lines(plan_lines, executed_instructions):
    if executed_instructions <= 0:
        return list(plan_lines)
    offset = executed_instructions * PLAN_STEP_INCREMENT
    shifted = []
    for line in plan_lines:
        if ":" not in line:
            shifted.append(line)
            continue
        time_part, rest = line.split(":", 1)
        try:
            new_time = float(time_part.strip()) + offset
        except ValueError:
            shifted.append(line)
            continue
        shifted.append(f"{new_time:.5f}: {rest.strip()}")
    return shifted


def extract_problem_number(filename):
    match = re.search(r'problem_(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    else:
        print(f"[WARNING] Filename does not match expected pattern: {filename}")
        return float('inf')  # Or another fallback value
 
 
def request_to_the_model(domain, problem, execution_times):
    url = "http://127.0.0.1:5000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer yourPassword123"
    }
    domain_problem = "\n".join([domain, problem])
 
    data = {
        "mode": "instruct",
        "stream": True,
        "character": "Example",
        "temperature": 0.01,
        "max_tokens": 2048,
        #"stop": ["\nUser:", "###", "<|endoftext|>", "END"],  # Stop sequences
        "messages": [
            #{"role": "system", "content": "return only the pddl plan and nothing else. use the correct format 0.00100: (action) /no_think"},
            {"role": "system", "content": """You are a PDDL planner. Return only the plan. Use the format: 
            action parameter1 ...
            action2 parameter1 parameter2 ...
            ...
            Finish each plan wiht 'END'. Do not include any explanations.  /no_think"""},
            #{"role": "system", "content": domain},
            {"role": "user", "content": domain_problem}
        ]
    }
 
    response = requests.post(url, headers=headers, json=data, stream=True)
 
    assistant_message = ''
    start = time.time()
 
    for line in response.iter_lines(decode_unicode=True):
        if line.strip() == "" or not line.startswith("data:"):
            continue
        data_str = line.lstrip("data:").strip()
        if 'END' in data_str:
            payload = json.loads(data_str)
            chunk = payload["choices"][0]["delta"].get("content", "")
            index = chunk.index('END') + len('END')
            chunk = chunk[:index]
            print(chunk)
            assistant_message += chunk
            break
        try:
            payload = json.loads(data_str)
            chunk = payload["choices"][0]["delta"].get("content", "")
            assistant_message += chunk
            print(chunk, end="", flush=True)
        except json.JSONDecodeError as e:
            print(f"\n[ERROR] Could not parse line: {line}\n{e}")
 
    print()
    print(f"Time required for the request: {time.time() - start:.2f}")
 
    lines = []
    ts = 0.00100
    increment = 0.00200
    for i, l in enumerate(assistant_message.split('\n')):
        l = l.strip()
        if len(l) > 0 and l!='END':
            l = f'{i*increment + ts:.5f}: ({l})'
        lines.append(l)
    execution_times.append(time.time() - start)
    assistant_message = '\n'.join(lines)
    print(assistant_message)
    return assistant_message, execution_times
 
def main(args):
    """
    Main function to orchestrate the loading of the model, processing of domains and problems,
    and validation of generated plans.
 
    Args:
        args (Namespace): Command line arguments containing dataset directory path,
                          validation tool path, and model name.
    """
    # Create folders structure
    folders_structure =  Folders_Structure(args.dataset_dir_path, args.model_name)
    test_set_json, domains_dir_path = folders_structure.create_structure()
 
    # Read JSON file and split domains and problems
    json_data = open_json_file(test_set_json)
    domains = [item['instruction'] for item in json_data]
    problems = [item['input'] for item in json_data]
 
    # Change into Str for api requests
    domains = list(map(str, domains))
    problems = list(map(str, problems))

    # Initialize trackers
    failed_problems = []
    error_counts = defaultdict(int)
    solved_by_attempt = defaultdict(int)
    solved_plans = 0

    # Initialize execution_times list
    execution_times = []

    val_timeout = getattr(args, "val_timeout", VAL_TIMEOUT_DEFAULT)
    val_bin_dir = None
    if args.validate_path:
        validate_path = Path(args.validate_path)
        val_bin_dir = validate_path if validate_path.is_dir() else validate_path.parent

    start_planning_time = time.time()
    plans_dir_path = None

    for i in range(len(problems)):
        print(i)
        # Write each domain and problem as a file
        domain = domains[i]
        domain_dir_path, domain_name, _domain_file_path, logs_dir_path = write_domain_and_create_logs_dir(domains_dir_path, domain)

        problem = problems[i]
        problem_name, problem_file_path = write_problem(problem, domain_name, domain_dir_path)

        current_problem_text = problem
        plan_lines = []
        executed_instructions = 0
        attempt = 0
        solved = False

        while attempt < MAX_RETRIES:
            attempt += 1
            print(f"[INFO] Attempt {attempt} for {problem_name}")
            assistant_message, execution_times = request_to_the_model(domain, current_problem_text, execution_times)

            new_plan_lines = sanitize_plan_lines(assistant_message)
            if not new_plan_lines:
                error_counts["empty_plan"] += 1
                print(f"[WARNING] Received empty plan on attempt {attempt} for {problem_name}")
                continue

            shifted_lines = shift_plan_lines(new_plan_lines, executed_instructions)
            candidate_lines = plan_lines + shifted_lines
            candidate_plan_text = build_plan_text(candidate_lines)

            if not candidate_plan_text:
                error_counts["empty_candidate_plan"] += 1
                print(f"[WARNING] Empty candidate plan after combining segments for {problem_name}")
                continue

            try:
                evaluation = validate_plan_text(
                    domain,
                    current_problem_text,
                    candidate_plan_text,
                    timeout=val_timeout,
                    val_bin_dir=val_bin_dir,
                )
            except Exception as exc:
                error_key = exc.__class__.__name__
                error_counts[error_key] += 1
                print(f"[ERROR] Validation failed on attempt {attempt} for {problem_name}: {exc}")
                break

            plan_lines = sanitize_plan_lines(evaluation.good_plan_prefix or candidate_plan_text)
            executed_instructions = evaluation.executed_instructions or len(plan_lines)

            if evaluation.status == "plan_valid":
                solved = True
                plans_dir_path = plans_dir_path or domain_dir_path
                print(f"[INFO] Plan valid for {problem_name} on attempt {attempt}.")
                break

            error_counts[evaluation.status] += 1
            correct_prefix = evaluation.executed_instructions or 0
            print(f"[INFO] Attempt {attempt} for {problem_name} ended with status {evaluation.status}.")
            print(f"[INFO] Valid prefix length: {correct_prefix} lines.")

            if evaluation.updated_problem_text:
                current_problem_text = evaluation.updated_problem_text

            if not plan_lines:
                executed_instructions = 0

        if solved:
            solved_plans += 1
            solved_by_attempt[attempt] += 1
            plan_text_to_save = build_plan_text(plan_lines)
            labelled_problem_name = f"{attempt}_{problem_name}"
            _plan_file_path, plans_dir_path = write_plan(domain_dir_path, labelled_problem_name, plan_text_to_save)
        else:
            failed_problems.append(Path(problem_file_path).name)
            if attempt >= MAX_RETRIES:
                error_counts["max_retries_reached"] += 1
                print(f"[WARNING] Maximum retries reached for {problem_name} without a valid plan.")
            else:
                error_counts["unsolved_before_max_retries"] += 1
                print(f"[WARNING] Aborting replanning early for {problem_name}.")

    full_planning_time = time.time() - start_planning_time
    minutes, seconds = divmod(full_planning_time, 60)
    hours, minutes = divmod(minutes, 60)
 
    # Save execution_times
    pickle_file_path = os.path.join(logs_dir_path, "execution_times.pkl")
    with open(pickle_file_path, "wb") as file:
        pickle.dump(execution_times, file)
    print("Times saved")
 
    # Compute statistics
    avg_time, min_time, max_time, median_time, std_dev = calculate_statistics(execution_times)
 
    # Sort the list based on the numeric part of the file name
    #failed_problems = sorted(
    #    failed_problems,
    #    key=lambda x: int(re.search(r'problem_(\d+)', os.path.basename(x)).group(1))
    #)
 
    failed_problems = sorted(failed_problems, key=extract_problem_number)

    # Print failed problems for invalid plans
    print("\nThe following problems did not result in viable plans:")
    for problem in failed_problems:
        print(problem)

    plans_failed = len(failed_problems)
    if plans_dir_path is None:
        plans_dir_path = domain_dir_path

    # Save a log file
    write_log(logs_dir_path, plans_failed, plans_dir_path, avg_time, min_time, max_time, median_time, std_dev, hours, minutes, seconds)

    # Print the output with several informations
    final_output(plans_failed, plans_dir_path, avg_time, min_time, max_time, median_time, std_dev)
    print(f"Planning time: {full_planning_time:.2f}")
    print(f"Hours: {int(hours)}, minutes: {int(minutes)}, seconds:{int(seconds)}")

    print(f"\nTotal solved plans: {solved_plans}")
    if solved_by_attempt:
        print("Solved plan distribution by attempt:")
        for attempt_count in sorted(solved_by_attempt):
            print(f"  Attempt {attempt_count}: {solved_by_attempt[attempt_count]}")

    if error_counts:
        print("\nValidation error counts:")
        for error_name, count in sorted(error_counts.items()):
            print(f"  {error_name}: {count}")
 
if __name__ == "__main__":
    # Command line argument parser setup
    parser = argparse.ArgumentParser(description='Process plans from the model')
    parser.add_argument('-d', '--dataset_dir_path', type=str, help='Dataset directory path')
    parser.add_argument('-v', '--validate_path', type=str, help='Validate tool path or VAL bin directory')
    parser.add_argument('--val-timeout', type=int, default=VAL_TIMEOUT_DEFAULT, help='Timeout (s) for VAL commands')
    parser.add_argument('-m', '--model_name', type=str, help='Name of the model used')
    args = parser.parse_args()
 
    # Launch main function
    main(args)
