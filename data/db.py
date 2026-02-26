import argparse
import sqlite3
import subprocess
import time
from data import DB_PATH, DB_SCHEMA_PATH

def _insert_row(table_name, columns, values, db=DB_PATH, timeout=30, retries=4):
    placeholders = ", ".join(["?"] * len(columns))
    col_list = ", ".join(columns)
    query = f"INSERT OR REPLACE INTO {table_name} ({col_list}) VALUES ({placeholders})"

    for attempt in range(retries):
        try:
            with sqlite3.connect(db, timeout=timeout) as c:
                c.isolation_level = None  # Autocommit mode
                c.execute(query, values)
            return
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < retries - 1:
                time.sleep(0.5 * (2 ** attempt)) # Exponential backoff: 0.5s, 1s, 2s，4s
            else:
                raise e

def insert_sigma0(net: str, img: int, min_eps: int, true_label=-1, adversarial_label=-1, total_duration=0.0, seed=0, db=DB_PATH):
    columns = ["net", "img", "min_eps", "true_label", "adversarial_label", "total_duration", "seed"]
    values = [net, img, min_eps, true_label, adversarial_label, total_duration, seed]
    _insert_row("sigma0", columns, values, db=db)

def sigma0_min_eps(net: str, img: int, db=DB_PATH) -> int | None:
    query = "SELECT min_eps FROM sigma0 WHERE net = ? AND img = ?"
    with sqlite3.connect(db) as c:
        result = c.execute(query, (net, img)).fetchone()
    return result[0] if result else None

def insert_lirpa(net: str, img: int, eps: int, cc=True, verified=False, timed_out=False, total_duration=0.0, db=DB_PATH):
    columns = ["net", "img", "eps", "cc", "verified", "timed_out", "total_duration", "skip"]
    values = [net, img, eps, int(cc), int(verified), int(timed_out), total_duration, 1]
    _insert_row("lirpa", columns, values, db=db)
    if not verified:
        with sqlite3.connect(db) as conn:
            result = conn.execute("SELECT max(eps) FROM lirpa_config WHERE net = ?", (net,)).fetchone()
        if result and result[0] is not None:
            max_eps = result[0]
            for higher_eps in range(eps + 1, max_eps + 1):
                values = [net, img, higher_eps, int(cc), 0, int(timed_out), 0.0, 1]
                _insert_row("lirpa", columns, values, db=db)

def insert_milp(net: str, img: int, eps: int, cc=True, verified=False, timed_out=False, found_adv=False, total_duration=0.0, db=DB_PATH):
    columns = ["net", "img", "eps", "cc", "verified", "timed_out", "found_adv", "total_duration", "skip"]
    values = [net, img, eps, int(cc), int(verified), int(timed_out), int(found_adv), total_duration, 1]
    _insert_row("milp", columns, values, db=db)
    if not verified:
        with sqlite3.connect(db) as conn:
            result = conn.execute("SELECT max(eps) FROM milp_config WHERE net = ?", (net,)).fetchone()
        if result and result[0] is not None:
            max_eps = result[0]
            for higher_eps in range(eps + 1, max_eps + 1):
                values = [net, img, higher_eps, int(cc), 0, int(timed_out), int(found_adv), 0.0, 1]
                _insert_row("milp", columns, values, db=db)

def insert_coverd(net: str, img: int, eps: int, cc=True, verified=False, timed_out=False, found_adv=False, estimated_time=0.0, total_duration=0.0, planning_duration=0.0, db=DB_PATH):
    columns = ["net", "img", "eps", "cc", "verified", "timed_out", "found_adv", "estimated_time", "total_duration", "planning_duration", "skip"]
    values = [net, img, eps, int(cc), int(verified), int(timed_out), int(found_adv), estimated_time, total_duration, planning_duration, 1]
    _insert_row("coverd", columns, values, db=db)
    if not verified:
        with sqlite3.connect(db) as conn:
            result = conn.execute("SELECT max(eps) FROM coverd_config WHERE net = ?", (net,)).fetchone()
        if result and result[0] is not None:
            max_eps = result[0]
            for higher_eps in range(eps + 1, max_eps + 1):
                values = [net, img, higher_eps, int(cc), 0, int(timed_out), int(found_adv), 0.0, 0.0, 0.0, 1]
                _insert_row("coverd", columns, values, db=db)

def insert_ours(net: str, img: int, eps: int, cc=True, estimated_time=0.0, use_cvd=False, starting_level=-1, skip=True, verified=False, timed_out=False, found_adv=False, total_duration=0.0, db=DB_PATH):
    """skip: equivalent to actually ran"""
    if not skip:  # plan_only
        with sqlite3.connect(db) as conn:
            result = conn.execute("SELECT time_limit_seconds FROM ours_config WHERE net = ? AND eps = ?", (net, eps)).fetchone()
        if estimated_time > result[0] * 2 or verified: # estimated_timed_out skip or end_level 0 or 1 skip -> equivalent to actually ran
            skip = True
    _use_cvd = int(use_cvd) if cc else -1
    columns = ["net", "img", "eps", "cc", "estimated_time", "use_cvd", "starting_level", "skip", "verified", "timed_out", "found_adv", "total_duration"]
    values = [net, img, eps, int(cc), estimated_time, _use_cvd, starting_level, int(skip), int(verified), int(timed_out), int(found_adv), total_duration]
    _insert_row("ours", columns, values, db=db)
    if found_adv or not cc:
        with sqlite3.connect(db) as conn:
            result = conn.execute("SELECT max(eps) FROM ours_config WHERE net = ?", (net,)).fetchone()
        assert result and result[0] is not None
        max_eps = result[0]
        for higher_eps in range(eps + 1, max_eps + 1):
            values = [net, img, higher_eps, int(cc), 0, -1, -1, 1, 0, int(timed_out), int(found_adv), 0.0]
            _insert_row("ours", columns, values, db=db)

def _insert_ours_detail(net: str, img: int, eps: int, true_label=-1, chosen_geometry="", choosing_strategy_duration=0.0, planning_duration=0.0, iterating_covering_duration=0.0, summed_duration=0.0, handle_unk_combs_duration=0.0, waiting_milp_duration=0.0, proved_with_milp=0, waiting_for_milp=0, end_level=0, adversarial_pixel_values="", adversarial_label=-1, db=DB_PATH):
    columns = ["net", "img", "eps", "true_label", "chosen_geometry", "choosing_strategy_duration", "planning_duration", "iterating_covering_duration", "summed_duration", "handle_unk_combs_duration", "waiting_milp_duration", "proved_with_milp", "waiting_for_milp", "end_level", "adversarial_pixel_values", "adversarial_label"]
    values = [net, img, eps, true_label, chosen_geometry, choosing_strategy_duration, planning_duration, iterating_covering_duration, summed_duration, handle_unk_combs_duration, waiting_milp_duration, proved_with_milp, waiting_for_milp, end_level, adversarial_pixel_values, adversarial_label]
    _insert_row("ours_detail", columns, values, db=db)

def insert_strategies(net: str, img: int, eps: int, use_cvd=False, starting_level=-1, estimated_time=0.0, geometry="", db=DB_PATH):
    columns = ["net", "img", "eps", "use_cvd", "starting_level", "estimated_time", "geometry"]
    values = [net, img, eps, int(use_cvd), starting_level, estimated_time, geometry]
    _insert_row("candidate_strategies", columns, values, db=db)

def insert_geometries(net: str, img: int, eps: int, geometry="", estimated_time=0.0, num_blocks=0, mean_size=0.0, db=DB_PATH):
    columns = ["net", "img", "eps", "geometry", "estimated_time", "num_blocks", "mean_size"]
    values = [net, img, eps, geometry, estimated_time, num_blocks, mean_size]
    _insert_row("candidate_geometries", columns, values, db=db)

def store_ours_result(net: str, eps: int, img: int, result: dict, skip=True, db=DB_PATH):
    """Store image analysis result into database tables."""
    summary = result.get('analysis_summary', {})
    if not summary:
        insert_ours(net, img, eps, cc=False, total_duration=result.get('running_time', 0.0), db=db)
        return
    # Extract main result data
    verified = summary.get('verified', False)
    timed_out = summary.get('timed_out', False)
    found_adv = 'adversarial_results' in summary
    estimated_time = summary.get('estimated_verification_time', 0.0)
    total_duration = result.get('running_time', 0.0)
    true_label = result.get('true_label', -1)

    # Extract strategy info from best_strategy
    best_strategy = summary.get('best_strategy', [])
    use_cvd = best_strategy[1] == 'SCSaP'
    starting_level = summary.get('starting_level', -1)

    # Insert main result into ours table
    insert_ours(net, img, eps, cc=True, estimated_time=estimated_time, use_cvd=use_cvd, starting_level=starting_level, skip=skip, verified=verified, timed_out=timed_out, found_adv=found_adv, total_duration=total_duration, db=db)

    # Extract detail data
    choosing_strategy_duration = summary.get('choosing_strategy_duration', 0.0)
    planning_duration = summary.get('planning_duration', 0.0)
    iterating_covering_duration = summary.get('SCSaP', {}).get('iterating_covering_duration', 0.0)
    summed_duration = summary.get('SCSaP', {}).get('summed_duration', 0.0)
    handle_unk_combs_duration = summary.get('handle_unk_combs_duration', 0.0)
    waiting_milp_duration = summary.get('waiting_milp_duration', 0.0)
    proved_with_milp = summary.get('proved_with_milp', 0)
    waiting_for_milp = summary.get('waiting_for_milp', 0)
    end_level = summary.get('end_level', 0)

    # Extract adversarial info
    adversarial_pixel_values = ""
    adversarial_label = -1
    if found_adv:
        adv_results = summary.get('adversarial_results', {})
        adversarial_pixel_values = adv_results.get('adversarial_pixel_values', '')
        adversarial_label = adv_results.get('adversarial_label', -1)

    # Extract chosen geometry from coverd
    chosen_geometry = ""
    coverd_info = summary.get('SCSaP', {})
    chosen_geom_dict = coverd_info.get('chosen_geometry', {})
    if chosen_geom_dict:
        geom_key = list(chosen_geom_dict.keys())[0] if chosen_geom_dict.keys() else ""
        if geom_key.startswith('Geometry('):
            chosen_geometry = geom_key[9:-1]  # Extract content between Geometry( and )

    # Insert detail data into ours_detail table
    _insert_ours_detail(net, img, eps, true_label, chosen_geometry, choosing_strategy_duration, planning_duration, iterating_covering_duration, summed_duration, handle_unk_combs_duration, waiting_milp_duration, proved_with_milp, waiting_for_milp, end_level, adversarial_pixel_values, adversarial_label, db)

    # Insert candidate geometries
    for geometry in summary.get('candidate_geometries', []):
        g_geometry_str = list(geometry.keys())[0]
        g_geometry = g_geometry_str[9:-1] if g_geometry_str.startswith('Geometry(') else g_geometry_str
        g_info_str = geometry[g_geometry_str]
        g_num_blocks = int(g_info_str.split(" blocks")[0])
        g_mean_size = float(g_info_str.split("mean size ")[1])
        g_estimated_time = geometry.get("estimated_summed_duration", 0.0)
        insert_geometries(net, img, eps, g_geometry, g_estimated_time, g_num_blocks, g_mean_size, db)
    if 'chosen_geometries' in summary:
        strategy_geometries = {l:g[9:-1] for l,g in summary.get('chosen_geometries', {}).items()}
        # Insert candidate strategies
        for strategy in summary.get('candidate_strategies', []):
            s_use_cvd = strategy[1] == 'SCSaP'
            s_starting_level = strategy[2]
            s_estimated_time = strategy[0]
            s_geometry = strategy_geometries.get(s_starting_level, "") if s_use_cvd else ""
            insert_strategies(net, img, eps, s_use_cvd, s_starting_level, s_estimated_time, s_geometry, db)

def _make_should_skip(table: str):
    """Factory function to create should_skip checkers for different tables."""
    def should_skip(net: str, img: int, eps: int, db=DB_PATH) -> bool:
        query = f"SELECT skip FROM {table} WHERE net = ? AND img = ? AND eps = ?"
        with sqlite3.connect(db) as c:
            result = c.execute(query, (net, img, eps)).fetchone()
        return result and result[0] == 1
    return should_skip

ours_should_skip = _make_should_skip("ours")
milp_should_skip = _make_should_skip("milp")
lirpa_should_skip = _make_should_skip("lirpa")
coverd_should_skip = _make_should_skip("coverd")

def ours_config_consistent(net: str, eps: int, img: int, batch_size: int, gpu_num: int, timeout:int, rk: int, rl: int, db=DB_PATH) -> bool:
    query = "SELECT * FROM ours_config WHERE net = ? AND eps = ?"
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        result = c.execute(query, (net, eps)).fetchone()
    if not result:
        return False
    config = dict(result)
    return (config['num_img'] > img and
            config['batch_size'] == batch_size and
            config['gpu_num'] == gpu_num and
            config['time_limit_seconds'] == timeout and
            config['rk'] == rk and
            config['rl'] == rl)

def milp_config_consistent(net: str, eps: int, img: int, timeout:int, db=DB_PATH) -> bool:
    query = "SELECT * FROM milp_config WHERE net = ? AND eps = ?"
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        result = c.execute(query, (net, eps)).fetchone()
    if not result:
        return False
    config = dict(result)
    return (config['num_img'] > img and
            config['time_limit_seconds'] == timeout)

def lirpa_config_consistent(net: str, eps: int, img: int, method: str, timeout:int, max_iteration:int, db=DB_PATH) -> bool:
    query = "SELECT * FROM lirpa_config WHERE net = ? AND eps = ?"
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        result = c.execute(query, (net, eps)).fetchone()
    if not result:
        return False
    config = dict(result)
    return (config['num_img'] > img and
            config['method'] == method and
            config['time_limit_seconds'] == timeout and
            config['max_iteration'] == max_iteration)

def coverd_config_consistent(net: str, eps: int, img: int, gpu_num: int, timeout:int, db=DB_PATH) -> bool:
    query = "SELECT * FROM coverd_config WHERE net = ? AND eps = ?"
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        result = c.execute(query, (net, eps)).fetchone()
    if not result:
        return False
    config = dict(result)
    return (config['num_img'] > img and
            config['gpu_num'] == gpu_num and
            config['time_limit_seconds'] == timeout)

def main():
    parser = argparse.ArgumentParser(description="DB helper")
    parser.add_argument("--database", "-db", metavar="FILE", help="Path to SQLite database file")
    parser.add_argument("--init", action="store_true", help="Initialize database schema")
    parser.add_argument("--select", '-s', metavar="COLUMNS", default="*", help="Columns to select")
    parser.add_argument("--from", '-f', dest="from_table", metavar="TABLE", help="Table to select from")
    parser.add_argument("--where", '-w', metavar="CLAUSE", help="Optional WHERE clause")
    parser.add_argument("--order-by", '-ob', metavar="CLAUSE", help="Optional ORDER BY clause")
    parser.add_argument("--query", '-q', metavar="QUERY", help="Custom SQL query to execute")
    parser.add_argument("--limit", '-l', type=int, metavar="NUM", help="Limit number of results")
    parser.add_argument("--drop", '-d', metavar="TABLE", help="Drop table or delete rows (use with -w to delete rows)")
    parser.add_argument("--export", '-e', metavar="FILE", help="Export rows to SQL file with INSERT OR REPLACE statements (use with -f and optionally -w)")
    args = parser.parse_args()

    db_path = args.database or DB_PATH

    if args.init:
        with sqlite3.connect(db_path) as c:
            c.executescript(open(DB_SCHEMA_PATH).read())
        return None

    if args.drop:
        if args.where:
            query = f"DELETE FROM {args.drop} WHERE {args.where}"
        else:
            query = f"DROP TABLE IF EXISTS {args.drop}"
        print(f"WARNING: About to execute: {query}")
        confirm = input("Are you sure? [(y)es/(n)o]: ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return None
        with sqlite3.connect(db_path) as conn:
            conn.execute(query)
            conn.commit()
        print(f"Executed: {query}")
        return None

    if args.export:
        if not args.from_table:
            parser.error("--export requires --from (-f) to specify table")

        # Build sqlite3 command with .mode insert
        sql_commands = [
            f".mode insert {args.from_table}",
            f".output {args.export}",
            f"SELECT * FROM {args.from_table}",
        ]

        if args.where:
            sql_commands[-1] += f" WHERE {args.where}"
        sql_commands.append("")  # Empty line to end

        sql_input = "\n".join(sql_commands)

        subprocess.run(['sqlite3', db_path], input=sql_input.encode(), check=True)

        # Count rows for feedback
        count_query = f"SELECT COUNT(*) FROM {args.from_table}"
        if args.where:
            count_query += f" WHERE {args.where}"

        with sqlite3.connect(db_path) as conn:
            count = conn.execute(count_query).fetchone()[0]

        print(f"Exported {count} rows from {args.from_table} to {args.export}")
        return None

    if args.from_table:
        query = f"SELECT {args.select} FROM {args.from_table}"
        if args.where:
            query += f" WHERE {args.where}"
        if args.order_by:
            query += f" ORDER BY {args.order_by}"
        if args.limit is not None:
            query += f" LIMIT {args.limit}"

        subprocess.run(['sqlite3', db_path, '-header', '-column', '-nullvalue', 'NULL', query])
        return None

    if args.query:
        subprocess.run(['sqlite3', db_path, '-header', '-column', '-nullvalue', 'NULL', args.query])
        return None

    if args.select and args.select != "*":
        parser.error("--select requires --from")

    parser.print_help()
    return None

if __name__ == "__main__":
    main()