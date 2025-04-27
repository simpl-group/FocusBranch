from json import load
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the increamental verification time comparison given result files.",
    )
    parser.add_argument("-d", "--data-dir", type=str, default=".")
    parser.add_argument("-rfs", '--result-files', type=str, nargs='+', required=True, help="The result files to be compared.")
    parser.add_argument("--log", action="store_true", help="Set x-axis to log scale.")
    parser.add_argument("-to", "--timeout", type=int, default=180)
    parser.add_argument("-o", "--output", type=str, default="increamental_time_comparison.pdf")
    parser.add_argument("--title", type=str, default="Increamental Verification Time")
    args = parser.parse_args()

    min_time = float('inf')
    last_ys = []
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for color, rf in zip(colors, args.result_files):
        if rf.endswith(".json"):
            method = rf[:-5]
        else:
            method = rf
            rf += ".json"
        assert os.path.exists(Path(args.data_dir)/rf), f"File {rf} does not exist in {args.data_dir}."

        with open(Path(args.data_dir)/rf, "r") as f:
            record: list[tuple[bool, float]] = load(f)

        total = len(record)

        # Filter and extract the time values for successful verifications, then sort them
        time_used = sorted(x[1] for x in record if x[0])

        last_x, last_y = time_used[-1], len(time_used)
        last_ys.append(last_y)
        if min_time > time_used[0]:
            min_time = time_used[0]

        ax.scatter(time_used, range(1, len(time_used) + 1), label=method, color=color, alpha=0.5, marker='.', s=20)
        ax.plot(time_used, range(1, len(time_used) + 1), color=color, alpha=0.7, linestyle='-', linewidth=1)
        ax.text(x=args.timeout, y=last_y, s=f"{last_y / total * 100:.1f}%", va='center', color=color)

    ax.set_ylim(bottom=0)
    plt.ylabel("# Verified images")
    ax.set_xlim(right=args.timeout)
    if args.log:
        ax.set_xlim(left=min_time)
        plt.xscale('log', base=10)
    else:
        ax.set_xlim(left=-1)
    plt.xlabel("Time (s)")
    plt.yticks(np.linspace(1, total, num=6, dtype=int).tolist()+last_ys)
    plt.grid()
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()
    if not args.output.endswith(".pdf"):
        args.output += ".pdf"
    plt.savefig(args.output, format='pdf')