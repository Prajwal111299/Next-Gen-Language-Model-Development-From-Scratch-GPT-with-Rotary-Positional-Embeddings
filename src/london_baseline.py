# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dev_path", help="Path to the birth dev TSV file")
    args = parser.parse_args()
    
    # Read the dev file.
    with open(args.dev_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total = len(lines)
    correct = 0

    # For each line, split on tab to grab the gold birthplace,
    # and compare (case-insensitively) against the baseline prediction "London".
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        gold = parts[1].strip()
        if gold.lower() == "london":
            correct += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0
    print(f"Baseline London accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
