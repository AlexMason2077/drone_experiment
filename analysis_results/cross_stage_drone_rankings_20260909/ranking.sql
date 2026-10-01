WITH values_by_mode AS (
 SELECT source,experiment_id,drone_id,battery_id,compare_stage,'raw' AS mode,
        raw_medium AS medium_value,raw_stage AS stage_value FROM five_drone_pairs
 UNION ALL
 SELECT source,experiment_id,drone_id,battery_id,compare_stage,'Bideal' AS mode,
        ideal_medium AS medium_value,ideal_stage AS stage_value FROM five_drone_pairs
)
SELECT *,
 RANK() OVER (PARTITION BY source,compare_stage,mode ORDER BY medium_value DESC) AS medium_rank,
 RANK() OVER (PARTITION BY source,compare_stage,mode ORDER BY stage_value DESC) AS stage_rank
FROM values_by_mode ORDER BY compare_stage,source,mode,drone_id;
