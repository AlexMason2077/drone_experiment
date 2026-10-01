WITH rates AS (
 SELECT *, 'endpoint' AS method, 60.0*(soc_start-soc_end)/ROUND(end_s-start_s,9) AS rate FROM windows
 UNION ALL
 SELECT *, 'ols' AS method, ols_rate AS rate FROM windows
), pairs AS (
 SELECT a.*, b.rate AS medium_rate, b.start_s AS medium_start_s, b.end_s AS medium_end_s
 FROM rates a JOIN rates b
 ON a.source=b.source AND a.drone_id=b.drone_id AND a.battery_id=b.battery_id
 AND a.threshold_pp=b.threshold_pp AND a.policy=b.policy AND a.method=b.method
 WHERE a.stage IN ('high','low') AND b.stage='medium' AND a.wind_level>0
), complete AS (
 SELECT *, COUNT(*) OVER (PARTITION BY source,stage,threshold_pp,policy,method) AS fleet_count
 FROM pairs
)
SELECT *, RANK() OVER (PARTITION BY source,stage,threshold_pp,policy,method ORDER BY rate DESC) AS stage_rank,
 RANK() OVER (PARTITION BY source,stage,threshold_pp,policy,method ORDER BY medium_rate DESC) AS medium_rank
FROM complete WHERE fleet_count=5
ORDER BY threshold_pp,policy,method,stage,source,drone_id;
