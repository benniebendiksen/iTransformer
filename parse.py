import re
import csv
import os
from collections import defaultdict

def parse_transformer_output(file_paths, output_file):
    # Check if output file exists to determine if we need to write headers
    file_exists = os.path.isfile(output_file)

    # Create a list to store all results for terminal display
    all_results = []

    # Open output file for appending
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = [
            'file', 'test_id', 'true_positives', 'true_negatives', 'false_positives', 'false_negatives',
            'accuracy', 'precision', 'recall', 'f1', 'pos_accuracy', 'neg_accuracy', 'neg_win_rate'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Process each file
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            print(f"\nProcessing file: {file_name}")

            # Create a list to store results for this file
            results_for_display = []

            # Read the input file, located in sub-directory slurm_outputs
            with open(os.path.join("slurm_outputs", file_path), 'r') as file:
                content = file.read()

            # Find all test blocks
            pattern = r'>>>>>>>testing : (.*?)<<<<<<.*?Confusion Matrix Breakdown:.*?True Positives.*?: (\d+).*?True Negatives.*?: (\d+).*?False Positives.*?: (\d+).*?False Negatives.*?: (\d+).*?Classification Performance:.*?Accuracy: ([\d.]+)%, Precision: ([\d.]+)%, Recall: ([\d.]+)%, F1: ([\d.]+)%'

            matches = re.findall(pattern, content, re.DOTALL)

            # Process and write each match
            for match in matches:
                test_id = match[0]
                true_positives = int(match[1])
                true_negatives = int(match[2])
                false_positives = int(match[3])
                false_negatives = int(match[4])
                accuracy = float(match[5])
                precision = float(match[6])
                recall = float(match[7])
                f1 = float(match[8])

                # Calculate positive accuracy: percentage of positive predictions that were correct
                # This is TP / (TP + FP), which is the same as precision
                total_pos_predictions = true_positives + false_positives
                pos_accuracy = (true_positives / total_pos_predictions * 100) if total_pos_predictions > 0 else 0

                # Calculate negative accuracy: percentage of negative predictions that were correct
                # This is TN / (TN + FN)
                total_neg_predictions = true_negatives + false_negatives
                neg_accuracy = (true_negatives / total_neg_predictions * 100) if total_neg_predictions > 0 else 0

                # Calculate negative win rate: TN / (TN + FN) as a ratio
                neg_win_rate = (true_negatives / false_negatives) if false_negatives > 0 else float('inf')

                # Extract window value for easier reading
                window_match = re.search(r'window_(\d+)', test_id)
                window = window_match.group(1) if window_match else "?"
                # Extract projection value for easier reading
                proj_match = re.search(r'projection_(\d+)', test_id)
                projection = proj_match.group(1) if proj_match else "?"

                # Create a short ID for display
                short_id = f"window_{window}_proj_{projection}"

                # Format the negative win rate for display
                if neg_win_rate == float('inf'):
                    neg_win_rate_display = "∞"
                else:
                    neg_win_rate_display = f"{neg_win_rate:.1f}"

                # Add to display results
                result_data = {
                    'file': file_name,
                    'id': short_id,
                    'window': window,
                    'projection': projection,
                    'tp': true_positives,
                    'tn': true_negatives,
                    'fp': false_positives,
                    'fn': false_negatives,
                    'acc': f"{accuracy:.1f}%",
                    'prec': f"{precision:.1f}%",
                    'rec': f"{recall:.1f}%",
                    'f1': f"{f1:.1f}%",
                    'pos_acc': f"{pos_accuracy:.1f}%",
                    'neg_acc': f"{neg_accuracy:.1f}%",
                    'neg_win': neg_win_rate_display
                }

                results_for_display.append(result_data)
                all_results.append(result_data)

                # Write to CSV
                writer.writerow({
                    'file': file_name,
                    'test_id': test_id,
                    'true_positives': true_positives,
                    'true_negatives': true_negatives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'pos_accuracy': pos_accuracy,
                    'neg_accuracy': neg_accuracy,
                    'neg_win_rate': neg_win_rate if neg_win_rate != float('inf') else 'inf'
                })

            # Group results by projection number for this file
            grouped_results = defaultdict(list)
            for result in results_for_display:
                grouped_results[result['projection']].append(result)

            # Display results grouped by projection
            for projection in sorted(grouped_results.keys()):
                print(f"\n=== FILE: {file_name} | PROJECTION {projection} ===")
                print_table(grouped_results[projection])

    # Now group all results by projection number across all files
    print("\n\n=== COMBINED RESULTS ACROSS ALL FILES ===")
    all_grouped_results = defaultdict(list)
    for result in all_results:
        all_grouped_results[result['projection']].append(result)

    # Display combined results grouped by projection
    for projection in sorted(all_grouped_results.keys()):
        print(f"\n=== COMBINED | PROJECTION {projection} ===")
        print_table(all_grouped_results[projection])

    print(f"\nParsing complete. Results appended to {output_file}")

def print_table(data):
    """Print a table to the terminal without external dependencies"""
    if not data:
        return

    # Get all keys for display
    if 'file' in data[0]:
        headers = ['file', 'window', 'tp', 'tn', 'fp', 'fn', 'acc', 'pos_acc', 'neg_acc', 'neg_win', 'prec', 'rec',
                   'f1']
        header_names = {
            'file': 'File', 'window': 'Window', 'tp': 'TP', 'tn': 'TN', 'fp': 'FP', 'fn': 'FN',
            'acc': 'ACC', 'pos_acc': 'POS ACC', 'neg_acc': 'NEG ACC', 'neg_win': 'NEG WIN',
            'prec': 'PREC', 'rec': 'REC', 'f1': 'F1'
        }
    else:
        headers = ['window', 'tp', 'tn', 'fp', 'fn', 'acc', 'pos_acc', 'neg_acc', 'neg_win', 'prec', 'rec', 'f1']
        header_names = {
            'window': 'Window', 'tp': 'TP', 'tn': 'TN', 'fp': 'FP', 'fn': 'FN',
            'acc': 'ACC', 'pos_acc': 'POS ACC', 'neg_acc': 'NEG ACC', 'neg_win': 'NEG WIN',
            'prec': 'PREC', 'rec': 'REC', 'f1': 'F1'
        }

    # Calculate column widths
    col_widths = {}
    for header in headers:
        col_widths[header] = len(header_names[header])
        for row in data:
            if header in row:  # Make sure the header exists in the row
                col_widths[header] = max(col_widths[header], len(str(row[header])))

    # Print header
    header_line = ' | '.join(header_names[h].ljust(col_widths[h]) for h in headers)
    print(header_line)
    print('-' * len(header_line))

    # Print rows
    for row in data:
        row_str = ' | '.join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers)
        print(row_str)

def main():
    file_paths = ["itransformer_434780_0.out", "itransformer_434780_1.out"]
    output_file = "transformer_results_combined.csv"
    parse_transformer_output(file_paths, output_file)

if __name__ == "__main__":
    main()
