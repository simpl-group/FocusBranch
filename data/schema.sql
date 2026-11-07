CREATE TABLE IF NOT EXISTS mean_concretecheck_time (
    net TEXT, -- contains num_pixels info, so we don't need to store num_pixels separately
    batch_size INTEGER,
    eps INTEGER,
    level INTEGER,
    mean_time REAL,
    PRIMARY KEY (net, batch_size, eps, level)
);

CREATE TABLE IF NOT EXISTS generator_speed (
    num_workers INTEGER,
    num_pixels INTEGER,
    generator_launch_time REAL,
    skipping_iter_speed REAL,
    normal_iter_speed REAL,
    PRIMARY KEY (num_workers, num_pixels)
);

CREATE TABLE IF NOT EXISTS SaP_speed (
    chunk_size INTEGER,
    num_indices INTEGER,
    level INTEGER,
    splitting_speed REAL,
    pruning_speed REAL,
    PRIMARY KEY (chunk_size, num_indices, level)
);

CREATE TABLE IF NOT EXISTS sigma0 (
    net TEXT,
    img INTEGER,
    min_eps INTEGER DEFAULT 500,
    total_duration REAL DEFAULT 0,
    true_label INTEGER DEFAULT -1,
    adversarial_label INTEGER DEFAULT -1,
    seed INTEGER DEFAULT 0,
    PRIMARY KEY (net, img)
);

CREATE TABLE IF NOT EXISTS ours_config (
    net TEXT,
    start_eps INTEGER,  -- the eps verified, start_eps..end_eps (inclusive)
    end_eps INTEGER,
    start_img INTEGER,  -- the images verified, start_img..end_img (inclusive)
    end_img INTEGER,
    num_gpus INTEGER,
    num_milp INTEGER,
    time_limit_seconds INTEGER DEFAULT 1800,
    rep_each_k INTEGER,
    rep_each_level INTEGER,
    concretecheck_batch INTEGER DEFAULT 0,  -- 0: any
    supercheck_batch INTEGER DEFAULT 0,  -- 0: any
    pool_factor INTEGER DEFAULT 0, -- pool size = supercheck_batch * pool_factor; 0: any
    probe_thickness INTEGER DEFAULT 50,
    PRIMARY KEY (net)
);

-- DROP TABLE IF EXISTS ours;
CREATE TABLE IF NOT EXISTS ours (
    net TEXT,
    img INTEGER,
    eps INTEGER,
    cc INTEGER DEFAULT 1,
    estimated_time REAL DEFAULT 0,
    use_cvd INTEGER DEFAULT 0,
    starting_level INTEGER DEFAULT -1,
    verified INTEGER DEFAULT 0,
    timed_out INTEGER DEFAULT 0,
    found_adv INTEGER DEFAULT 0,
    total_duration REAL DEFAULT 0,
    skip INTEGER DEFAULT 0,
    tag TEXT DEFAULT '',  -- free label of the run
    PRIMARY KEY (net, img, eps)
);

CREATE TABLE IF NOT EXISTS ours_detail (
    net TEXT,
    img INTEGER,
    eps INTEGER,
    true_label INTEGER DEFAULT -1,
    chosen_geometry TEXT DEFAULT '',
    choosing_strategy_duration REAL DEFAULT 0,
    planning_duration REAL DEFAULT 0,
    iterating_covering_duration REAL DEFAULT 0,
    summed_duration REAL DEFAULT 0,
    handle_unk_combs_duration REAL DEFAULT 0,
    waiting_milp_duration REAL DEFAULT 0,
    proved_with_milp INTEGER DEFAULT 0,
    waiting_for_milp INTEGER DEFAULT 0,
    end_level INTEGER DEFAULT -1,
    adversarial_pixel_values TEXT DEFAULT '',
    adversarial_label INTEGER DEFAULT -1,
    PRIMARY KEY (net, img, eps)
);

DROP VIEW IF EXISTS ours_all;
CREATE VIEW ours_all AS
SELECT o.*, od.true_label, od.chosen_geometry, od.choosing_strategy_duration, od.planning_duration, od.iterating_covering_duration, od.summed_duration, od.handle_unk_combs_duration, od.waiting_milp_duration, od.proved_with_milp, od.waiting_for_milp, od.end_level, od.adversarial_pixel_values, od.adversarial_label
FROM ours o
LEFT JOIN ours_detail od
ON o.net = od.net AND o.img = od.img AND o.eps = od.eps;

DROP VIEW IF EXISTS ours_digest;
CREATE VIEW ours_digest AS
  WITH RECURSIVE nums(i) AS (
    SELECT 1
    UNION ALL
    SELECT i + 1 FROM nums WHERE i < 8
  ),
  sigma0_stats AS (
    SELECT
      net,
      i,
      SUM(min_eps <= i) AS Adv_sigma0
    FROM sigma0
    CROSS JOIN nums
    GROUP BY net, i
  )
  SELECT
    oa.net,
    SUM(cc = 1) AS "CC",
    eps,
    Adv_sigma0,
    SUM(found_adv = 1 AND oa.total_duration = 0) AS "Adv_known",
    SUM(found_adv = 1 AND oa.total_duration > 0) AS "Adv_ours",
    SUM(cc = 1) - SUM(found_adv = 1) AS "Cand",
    SUM(verified = 1) AS "R",
    SUM(timed_out = 1) AS "TO",
    ROUND((SUM(verified=1) + SUM(found_adv = 1 AND oa.total_duration > 0)) * 1.0 / (SUM(cc = 1) - SUM(found_adv = 1 AND oa.total_duration = 0)), 3) AS "NTO/Cand",
    CONCAT(100 * ROUND(SUM(verified=1) * 1.0 / (SUM(cc = 1) - SUM(found_adv = 1)), 3), '\%') AS "R/Cand",
    ROUND(AVG(CASE WHEN verified = 1 THEN oa.total_duration END), 4) AS "T_avg",
    ROUND(MAX(CASE WHEN verified = 1 THEN oa.total_duration END), 4) AS "T_max",
    SUM(verified = 1 AND starting_level < eps) AS "ls<e",
    SUM(verified = 1 AND starting_level < eps AND end_level > starting_level) AS "Branch_con",
    SUM(verified = 1 AND starting_level = eps) AS "ls=e",
    SUM(verified = 1 AND use_cvd = 0) AS "CCSaP",
    SUM(verified = 1 AND use_cvd = 1) AS "SCSaP",
    SUM(verified = 1 AND use_cvd = 1 AND starting_level = eps) AS "SC@ls=e",
    SUM(verified = 1 AND use_cvd = 1 AND starting_level < eps) AS "SC@ls<e"
  FROM sigma0_stats
  JOIN ours_all oa ON (oa.eps = sigma0_stats.i AND oa.net = sigma0_stats.net)
  GROUP BY oa.net, oa.eps
  HAVING MAX(oa.total_duration) > 0
  ORDER BY
    CASE oa.net
      WHEN 'afc1' THEN 1
      WHEN 'afc2' THEN 2
      WHEN 'afc3' THEN 3
      WHEN 'afc4' THEN 4
      WHEN 'afc5' THEN 5
      WHEN 'bfc1' THEN 6
      WHEN 'bcv1' THEN 7
      WHEN 'bcv2' THEN 8
      WHEN 'bcv3' THEN 9
      WHEN 'bcv4' THEN 10
      WHEN 'bcv6' THEN 11
      WHEN 'bcv7' THEN 12
      WHEN 'bcv5' THEN 13
      ELSE 100
    END, eps;

CREATE TABLE IF NOT EXISTS candidate_strategies (
    net TEXT,
    img INTEGER,
    eps INTEGER,
    starting_level INTEGER,
    use_cvd INTEGER DEFAULT 0,
    estimated_time REAL,
    geometry TEXT,
    PRIMARY KEY (net, img, eps, starting_level, use_cvd)
);

CREATE TABLE IF NOT EXISTS candidate_geometries (
    net TEXT,
    img INTEGER,
    eps INTEGER,
    geometry TEXT,
    estimated_time REAL,
    num_blocks INTEGER,
    mean_size REAL,
    PRIMARY KEY (net, img, eps, geometry)
);

CREATE TABLE IF NOT EXISTS lirpa_config (
    net TEXT,
    start_eps INTEGER,  -- inclusive ranges, as in ours_config
    end_eps INTEGER,
    start_img INTEGER,
    end_img INTEGER,
    method TEXT,
    time_limit_seconds INTEGER DEFAULT 1800,
    max_iteration INTEGER,
    PRIMARY KEY (net)
);

CREATE TABLE IF NOT EXISTS lirpa (
    net TEXT,
    img INTEGER,
    cc INTEGER DEFAULT 1,
    eps INTEGER,
    verified INTEGER DEFAULT 0,
    total_duration REAL DEFAULT 0,
    skip INTEGER DEFAULT 0,
    PRIMARY KEY (net, img, eps)
);

CREATE TABLE IF NOT EXISTS milp_config (
    net TEXT,
    start_eps INTEGER,  -- inclusive ranges, as in ours_config
    end_eps INTEGER,
    start_img INTEGER,
    end_img INTEGER,
    time_limit_seconds INTEGER DEFAULT 1800,
    PRIMARY KEY (net)
);

CREATE TABLE IF NOT EXISTS milp (
    net TEXT,
    img INTEGER,
    cc INTEGER DEFAULT 1,
    eps INTEGER,
    verified INTEGER DEFAULT 0,
    timed_out INTEGER DEFAULT 0,
    found_adv INTEGER DEFAULT 0,
    total_duration REAL DEFAULT 0,
    skip INTEGER DEFAULT 0,
    PRIMARY KEY (net, img, eps)
);

INSERT OR REPLACE INTO ours_config
(net, start_eps, end_eps, start_img, end_img, num_gpus, num_milp, time_limit_seconds, rep_each_k, rep_each_level, concretecheck_batch, supercheck_batch, pool_factor, probe_thickness) VALUES
('bfc1', 1, 8, 0, 99, 4, 5, 1800, 200, 5000, 4000, 800, 2, 50),
('afc5', 1, 8, 0, 99, 4, 5, 1800, 200, 5000, 4000, 800, 2, 50),
('afc1', 1, 8, 0, 99, 4, 5, 1800, 200, 5000, 4000, 800, 2, 50),
('afc2', 1, 8, 0, 99, 4, 5, 1800, 200, 5000, 4000, 800, 2, 50),
('afc3', 1, 8, 0, 99, 4, 5, 1800, 200, 5000, 4000, 800, 2, 50),
('afc4', 1, 8, 0, 99, 4, 5, 1800, 200, 5000, 4000, 800, 2, 50),
('bcv1', 1, 6, 0, 99, 4, 5, 1800, 200, 500, 0, 400, 2, 5),
('bcv2', 1, 6, 0, 99, 4, 5, 1800, 200, 500, 0, 400, 2, 5),
('bcv3', 1, 6, 0, 99, 4, 5, 1800, 200, 500, 0, 400, 2, 5),
('bcv4', 3, 3, 0, 9, 4, 5, 3600, 100, 100, 5, 5, 4, 1),
('bcv5', 3, 3, 0, 9, 4, 50, 1800, 200, 500, 0, 400, 2, 5),
('bcv6', 1, 6, 0, 99, 4, 5, 1800, 200, 500, 0, 400, 2, 5),
('bcv7', 1, 6, 0, 99, 4, 5, 1800, 200, 500, 0, 400, 2, 5);

INSERT OR REPLACE INTO lirpa_config (net, start_eps, end_eps, start_img, end_img, method, time_limit_seconds, max_iteration) VALUES
('bfc1', 1, 8, 0, 99, 'CROWN-Optimized', 1800, 1000),
('afc5', 1, 8, 0, 99, 'CROWN-Optimized', 1800, 1000),
('afc1', 1, 8, 0, 99, 'CROWN-Optimized', 1800, 1000),
('afc2', 1, 8, 0, 99, 'CROWN-Optimized', 1800, 1000),
('afc3', 1, 8, 0, 99, 'CROWN-Optimized', 1800, 1000),
('afc4', 1, 8, 0, 99, 'CROWN-Optimized', 1800, 1000);

INSERT OR REPLACE INTO milp_config (net, start_eps, end_eps, start_img, end_img, time_limit_seconds) VALUES
('bfc1', 1, 8, 0, 99, 1800),
('afc5', 1, 8, 0, 99, 1800),
('afc1', 1, 8, 0, 99, 1800),
('afc2', 1, 8, 0, 99, 1800),
('afc3', 1, 8, 0, 99, 1800),
('afc4', 1, 8, 0, 99, 1800),
('bcv1', 1, 6, 0, 99, 1800),
('bcv2', 1, 6, 0, 99, 1800),
('bcv3', 1, 6, 0, 99, 1800),
('bcv4', 3, 3, 0, 9, 3600),
('bcv5', 3, 3, 0, 9, 1800),
('bcv6', 1, 6, 0, 99, 1800),
('bcv7', 1, 6, 0, 99, 1800);
