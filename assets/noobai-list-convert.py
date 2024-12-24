#!/usr/bin/env python3

"""
MIT License

Copyright (c) 2024 itsjustregi (Regi E. regi@bynine.io)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import csv
import argparse
import sys
import time
from multiprocessing import Pool, cpu_count
import psutil
import os
from colorama import init, Fore
from tqdm import tqdm
from typing import Dict, List, Set
import inquirer

# Initialize Colorama for Windows compatibility
init(autoreset=True)


def get_optimal_chunk_size():
    available_memory = psutil.virtual_memory().available
    min_memory = 8 * 1024 * 1024 * 1024  # 8 GB
    max_memory = 64 * 1024 * 1024 * 1024  # 64 GB
    memory_to_use = max(min(available_memory * 0.75, max_memory), min_memory)
    return int(memory_to_use / (1024 * 1024))  # Convert to MB


def select_columns(headers):
    questions = [
        inquirer.Checkbox(
            "columns",
            message="Select columns to keep (use space to select, enter to confirm)",
            choices=headers,
            default=headers[:3],  # Default to first 3 columns
        ),
    ]

    answers = inquirer.prompt(questions)

    if not answers or not answers["columns"]:
        print(Fore.YELLOW + "No columns selected. Defaulting to all columns.")
        return headers

    return answers["columns"]


def process_chunk(chunk):
    result = {}
    for row in chunk:
        key = next(iter(row.values()))  # Use the first column as the key
        result[key] = {col: row[col] for col in row if col != key}
    return result


def merge_results(results):
    final_result = {}
    for result in results:
        for key, data in result.items():
            if key not in final_result:
                final_result[key] = {col: set() for col in data}
            for col, values in data.items():
                final_result[key][col].add(values)

    for key in final_result:
        for col in final_result[key]:
            final_result[key][col] = list(final_result[key][col])

    return final_result


def csv_to_dict(file_path, selected_columns, top_percent=0.12):
    start_time = time.time()
    process = psutil.Process(os.getpid())

    chunk_size = get_optimal_chunk_size()
    print(Fore.CYAN + f"Using chunk size of {chunk_size} MB")

    print(Fore.YELLOW + "Reading CSV file...")
    try:
        with open(file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            all_rows = list(reader)

            # Sort rows by the 'count' column in descending order
            all_rows.sort(key=lambda x: int(x["count"]), reverse=True)

            # Calculate the number of rows to keep
            rows_to_keep = int(len(all_rows) * top_percent)

            # Keep only the top rows
            top_rows = all_rows[:rows_to_keep]

            chunks = []
            current_chunk = []
            current_chunk_size = 0

            for row in tqdm(top_rows, desc="Processing top rows", unit="row"):
                filtered_row = {col: row[col] for col in selected_columns if col in row}
                current_chunk.append(filtered_row)
                current_chunk_size += sys.getsizeof(filtered_row)
                if (
                    current_chunk_size >= chunk_size * 1024 * 1024
                ):  # Convert MB to bytes
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_chunk_size = 0

            if current_chunk:
                chunks.append(current_chunk)

    except FileNotFoundError:
        print(Fore.RED + f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except csv.Error as e:
        print(Fore.RED + f"Error reading CSV file: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print(Fore.GREEN + f"CSV file read. Total chunks: {len(chunks)}")
    print(Fore.CYAN + f"RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    print(Fore.YELLOW + f"Processing data using {cpu_count()} cores...")
    with Pool(processes=cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_chunk, chunks),
                total=len(chunks),
                desc="Processing chunks",
            )
        )

    print(Fore.GREEN + "Merging results...")
    final_result = merge_results(results)

    end_time = time.time()
    print(Fore.GREEN + f"Processing completed in {end_time - start_time:.2f} seconds")
    print(
        Fore.CYAN + f"Final RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB"
    )

    return final_result


def save_dict_to_file(data, output_file, dict_name):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("from typing import Dict, List, Set\n\n")
            f.write(
                '"""\nThis file is used in the EasyNoobAI custom node for ComfyUI.\n'
            )
            f.write(
                'It contains a dictionary of characters with their associated triggers and core tags.\n"""\n\n'
            )
            f.write(f"{dict_name}: Dict[str, Dict[str, List[str]]] = ")
            f.write(str(data))
            print(Fore.GREEN + f"Dictionary saved to {output_file}")
    except IOError as e:
        print(
            Fore.RED + f"Error writing to file '{output_file}': {str(e)}",
            file=sys.stderr,
        )
        sys.exit(1)


def display_dict_sample(data: Dict[str, Dict[str, List[str]]], dict_name: str):
    print(f"\n{Fore.CYAN}Displaying sample of {dict_name}:")
    print(f"{Fore.YELLOW}First 13 entries:")
    for i, (key, value) in enumerate(list(data.items())[:13]):
        print(f"{Fore.GREEN}{i+1}. {Fore.RESET}{key}: {value}")

    print(f"\n{Fore.YELLOW}Last 13 entries:")
    for i, (key, value) in enumerate(list(data.items())[-13:], start=len(data) - 12):
        print(f"{Fore.GREEN}{i}. {Fore.RESET}{key}: {value}")

    print(f"\n{Fore.CYAN}Total entries in {dict_name}: {Fore.GREEN}{len(data)}")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV to Python dictionary")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file path")
    parser.add_argument("-o", "--output", required=True, help="Output Python file path")
    parser.add_argument(
        "-n", "--name", default="character_dict", help="Name of the output dictionary"
    )
    parser.add_argument(
        "-d",
        "--display",
        action="store_true",
        help="Display a sample of the dictionary after processing",
    )
    parser.add_argument(
        "-c", "--columns", nargs="+", help="Columns to keep (space-separated)"
    )
    parser.add_argument(
        "-t",
        "--top",
        type=float,
        default=0.025,
        help="Top percentage of entries to keep (default: 0.025)",
    )
    args = parser.parse_args()

    # Read CSV headers
    with open(args.input, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

    # If columns are not specified via CLI, use interactive selection
    if not args.columns:
        try:
            args.columns = select_columns(headers)
        except KeyboardInterrupt:
            print(Fore.RED + "\nColumn selection cancelled.")
            sys.exit(0)

    try:
        data = csv_to_dict(args.input, args.columns, args.top)
        save_dict_to_file(data, args.output, args.name)
        if args.display:
            display_dict_sample(data, args.name)
    except Exception as e:
        print(Fore.RED + f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
