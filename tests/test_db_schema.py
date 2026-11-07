"""The configuration tables (one row per network, inclusive ranges of eps and images) and the migration of databases
made with the older schema (renamed networks, ours.tag, no coverd tables). Run with `python -m pytest tests`."""
import sqlite3
from pathlib import Path

from data.db import migrate, ours_config_consistent, milp_config_consistent, lirpa_config_consistent, insert_ours

SCHEMA = (Path(__file__).resolve().parent.parent / "data" / "schema.sql").read_text()
RUN = dict(gpu_num=4, milp_num=5, timeout=1800, rep_each_k=200, rep_each_level=5000, concretecheck_batch=4000, supercheck_batch=800, pool_factor=2,
           probe_thickness=50)


def fresh(tmp_path) -> str:
    db = str(tmp_path / "new.db")
    with sqlite3.connect(db) as c:
        c.executescript(SCHEMA)
    return db


def test_config_ranges(tmp_path):
    db = fresh(tmp_path)
    assert ours_config_consistent("afc5", 5, 0, 99, **RUN, db=db) == (True, "")
    ok, why = ours_config_consistent("afc5", 9, 90, 100, **RUN, db=db)  # eps 1..8, images 0..99
    assert not ok and "-e 9" in why and "images 90..100" in why
    assert not ours_config_consistent("afc5", 5, 0, 99, **{**RUN, "gpu_num": 2}, db=db)[0]
    assert not ours_config_consistent("bafc1", 5, 0, 99, **RUN, db=db)[0]  # renamed
    assert not ours_config_consistent("afc5", 5, 0, 99, **{**RUN, "supercheck_batch": 400}, db=db)[0]
    conv = {**RUN, "rep_each_level": 500, "probe_thickness": 5, "supercheck_batch": 400}
    assert ours_config_consistent("bcv1", 6, 0, 99, **{**conv, "concretecheck_batch": 200}, db=db)[0]  # bcv1 takes any -cb
    assert milp_config_consistent("bcv4", 3, 0, 9, 3600, db=db) and not milp_config_consistent("bcv4", 2, 0, 9, 3600, db=db)
    assert not milp_config_consistent("bcv5", 3, 0, 10, 1800, db=db)  # 10 images
    assert lirpa_config_consistent("afc4", 8, 0, 99, "CROWN-Optimized", 1800, 1000, db=db)


def test_unsafe_image_fills_the_higher_eps_up_to_end_eps(tmp_path):
    db = fresh(tmp_path)
    insert_ours("afc1", 3, 6, found_adv=True, db=db)
    with sqlite3.connect(db) as c:
        assert [r[0] for r in c.execute("SELECT eps FROM ours WHERE net = 'afc1' AND img = 3 ORDER BY eps")] == [6, 7, 8]
        assert {r[0] for r in c.execute("SELECT tag FROM ours")} == {""}


def test_migrate_old_database(tmp_path):
    db = str(tmp_path / "old.db")
    with sqlite3.connect(db) as c:  # the tables as they were, with a few rows
        c.executescript("""
            CREATE TABLE ours (net TEXT, img INTEGER, eps INTEGER, cc INTEGER DEFAULT 1, estimated_time REAL DEFAULT 0, use_cvd INTEGER DEFAULT 0,
                starting_level INTEGER DEFAULT -1, verified INTEGER DEFAULT 0, timed_out INTEGER DEFAULT 0, found_adv INTEGER DEFAULT 0,
                total_duration REAL DEFAULT 0, skip INTEGER DEFAULT 0, PRIMARY KEY (net, img, eps));
            CREATE TABLE ours_config (net TEXT, eps INTEGER, gpu_num INTEGER, milp_num INTEGER, time_limit_seconds INTEGER, rep_each_k INTEGER,
                rep_each_level INTEGER, num_img INTEGER, concretecheck_batch INTEGER, supercheck_batch INTEGER, pool_size INTEGER,
                probe_thickness INTEGER, PRIMARY KEY (net, eps));
            CREATE TABLE milp_config (net TEXT, eps INTEGER, time_limit_seconds INTEGER, num_img INTEGER, PRIMARY KEY (net, eps));
            CREATE TABLE coverd (net TEXT, img INTEGER, eps INTEGER, PRIMARY KEY (net, img, eps));
            CREATE TABLE coverd_config (net TEXT, eps INTEGER, PRIMARY KEY (net, eps));
            CREATE TABLE mean_concretecheck_time (net TEXT, batch_size INTEGER, eps INTEGER, level INTEGER, mean_time REAL,
                PRIMARY KEY (net, batch_size, eps, level));
            INSERT INTO ours (net, img, eps, verified, total_duration) VALUES ('bafc1', 27, 5, 1, 9.5), ('fc4', 0, 4, 1, 1.0), ('afc1', 74, 4, 1, 469.0);
            INSERT INTO ours_config VALUES ('bafc1', 5, 4, 5, 1800, 200, 5000, 100, 4000, 800, 1600, 50);
            INSERT INTO milp_config VALUES ('bcv4', 1, 1800, 10), ('bcv4', 3, 3600, 10);
            INSERT INTO mean_concretecheck_time VALUES ('bafc1', 4000, 5, 2, 3.74e-5);
        """)
    migrate(db)
    with sqlite3.connect(db) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert "coverd" not in tables and "coverd_config" not in tables
        assert sorted(c.execute("SELECT net, img, eps, verified, total_duration, tag FROM ours")) == \
            [("afc1", 74, 4, 1, 469.0, ""), ("afc4", 0, 4, 1, 1.0, ""), ("afc5", 27, 5, 1, 9.5, "")]
        assert c.execute("SELECT net FROM mean_concretecheck_time").fetchall() == [("afc5",)]
        assert c.execute("SELECT start_eps, end_eps, start_img, end_img, time_limit_seconds FROM milp_config WHERE net = 'bcv4'").fetchall() == [(3, 3, 0, 9, 3600)]
        assert c.execute("SELECT count(*) FROM ours_config").fetchone()[0] == 13
    assert ours_config_consistent("afc5", 5, 0, 99, **RUN, db=db)[0]
    migrate(db)  # again: nothing changes
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT count(*) FROM ours").fetchone()[0] == 3
