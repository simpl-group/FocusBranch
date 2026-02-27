import argparse
from utils.shared import add_shared_arguments
from data.db import sigma0_min_eps, insert_ours
import sqlite3
from data import DB_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample-wise (local) L0 robustness verification for neural networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared_arguments(parser)
    parser.add_argument("--database", "-db", metavar="FILE", help="Path to SQLite database file")
    parser.add_argument("--unskip-higher-timed-out", action="store_true", help="Unskip higher eps if a lower eps is timed out")
    args = parser.parse_args()
    db_path = args.database or DB_PATH
    timed_out_count_adjust = [0] * (args.epsilon + 1)
    for e in range(1, args.epsilon+1):
        for i in range(args.start_idx, args.start_idx + args.test_batches):
            can_be_fixed = False
            with sqlite3.connect(db_path) as c:
                adv_result = c.execute(f"SELECT found_adv, total_duration FROM ours WHERE net = ? AND img = ? AND eps = ?", (args.net, i, e)).fetchone()
            adv_min_eps = sigma0_min_eps(net=args.net, img=i)
            if adv_min_eps is not None and adv_min_eps <= e:
                can_be_fixed = True
                if adv_result:
                    if adv_result[0] != 1:  # sigma0 found an adversarial example, but ours did not
                        print(f"Image {i:>2}, eps {e} already has a known adv example found by sigma0 with eps {adv_min_eps}")
                        if args.force_store:
                            print(f"Updating the record for image {i:>2}, eps {e} with the known adversarial example.")
                            insert_ours(net=args.net, img=i, eps=adv_min_eps, found_adv=True, use_cvd=-1, db=db_path)
                    elif adv_result[1] > 0: # both found an adversarial example, but ours took much longer time
                        print(f"Ours found adv for image {i:>2}, eps {e} with duration {adv_result[1]:>8.3f} s, while sigma0 also found with eps {adv_min_eps}")
                        print("    This should not happen, unless sigma0's result is not recorded in the database.")
                        if args.force_store:
                            print(f"Updating the record for image {i:>2}, eps {e} with the known adversarial example.")
                            insert_ours(net=args.net, img=i, eps=adv_min_eps, found_adv=True, use_cvd=-1, db=db_path)
            elif adv_result and adv_result[0] == 1:
                print(f"Image {i:>2}, eps {e} adv found by ours, but sigma0 found with eps {adv_min_eps} > {e}")
            with sqlite3.connect(db_path) as c:
                result = c.execute(f"SELECT timed_out, total_duration FROM ours WHERE net = ? AND img = ? AND eps = ?", (args.net, i, e)).fetchone()
            if result and result[0] == 1 and result[1] == 0:
                print(f"Image {i:>2}, eps {e} analysis was skipped due to a timed out in previous eps. Can be fixed by -fs: {can_be_fixed}")
                if can_be_fixed: timed_out_count_adjust[e] -= 1
                if args.unskip_higher_timed_out and not can_be_fixed:
                    print(f"Updating the record for image {i:>2}, eps {e} to set skip to 0")
                    insert_ours(net=args.net, img=i, eps=e, use_cvd=-1, timed_out=1, skip=False, db=db_path)
    if any(timed_out_count_adjust):
        print("Timed out count adjustment for each eps:")
        for e in range(1, args.epsilon+1):
            if timed_out_count_adjust[e] != 0:
                print(f"eps {e}: {timed_out_count_adjust[e]}")