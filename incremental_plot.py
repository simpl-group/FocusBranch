from json import load
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the increamental verification time comparison between two methods.",
    )
    parser.add_argument("-d", "--data-dir", type=str, default=".")
    parser.add_argument("-m1", "--method1", type=str, default="Ours")
    parser.add_argument("-m2", "--method2", type=str, default="LiRPA")
    parser.add_argument("--log", action="store_true", help="Set x-axis to log scale.")
    parser.add_argument("-to", "--timeout", type=int, default=180)
    parser.add_argument("-o", "--output", type=str, default="increamental_time_comparison.pdf")
    parser.add_argument("--title", type=str, default="Increamental Verification Time")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, f"{args.method1}.json"), "r") as f:
        record_m1: list[tuple[bool, float]] = load(f)
    with open(os.path.join(args.data_dir, f"{args.method2}.json"), "r") as f:
        record_m2: list[tuple[bool, float]] = load(f)

    assert len(record_m1) == len(record_m2), "The two records must have the same length."
    total = len(record_m2)
    assert all([record_m1[i][0] == record_m2[i][0] for i in range(total) if record_m2[i][0]])

    # Filter and extract the time values for successful verifications, then sort them
    m1_time_used = sorted(x[1] for x in record_m1 if x[0])
    m2_time_used = sorted(x[1] for x in record_m2 if x[0])

    m1_last_x, m1_last_y = m1_time_used[-1], len(m1_time_used)
    m2_last_x, m2_last_y = m2_time_used[-1], len(m2_time_used)
    min_time = min(m1_time_used[0], m2_time_used[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(m1_time_used, range(1, len(m1_time_used) + 1), label=args.method1, color='blue', alpha=0.5, marker='.', s=20)
    ax.scatter(m2_time_used, range(1, len(m2_time_used) + 1), label=args.method2, color='red', alpha=0.5, marker='.', s=20)

    ax.plot(m1_time_used, range(1, len(m1_time_used) + 1), color='blue', alpha=0.7, linestyle='-', linewidth=1)
    ax.plot(m2_time_used, range(1, len(m2_time_used) + 1), color='red', alpha=0.7, linestyle='-', linewidth=1)

    ax.text(x=args.timeout, y=m1_last_y, s=f"{m1_last_y / total * 100:.1f}%", va='center', color='blue')
    ax.text(x=args.timeout, y=m2_last_y, s=f"{m2_last_y / total * 100:.1f}%", va='center', color='red')

    ax.set_ylim(bottom=0)
    plt.ylabel("# Verified images")
    ax.set_xlim(right=args.timeout)
    if args.log:
        ax.set_xlim(left=min_time)
        plt.xscale('log', base=10)
    else:
        ax.set_xlim(left=-1)
    plt.xlabel("Time (s)")
    plt.yticks(np.linspace(1, total, num=6, dtype=int).tolist()+[m1_last_y, m2_last_y])
    plt.grid()
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, format='pdf')