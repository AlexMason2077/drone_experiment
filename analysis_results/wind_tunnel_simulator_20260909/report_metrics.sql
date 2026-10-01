
WITH per_curve_stage AS (
 SELECT source, drone_id, stage, COUNT(*) AS observed_soc_steps,
        SUM(duration_s) AS actual_time_s,
        SUM(predicted_dwell_s) AS predicted_time_s
 FROM holdout_predictions GROUP BY source, drone_id, stage
)
SELECT source,drone_id,stage,observed_soc_steps,actual_time_s,predicted_time_s,
       100.0*ABS(predicted_time_s-actual_time_s)/actual_time_s AS absolute_error_pct,
       60.0*observed_soc_steps/actual_time_s AS actual_rate_pp_min,
       60.0*observed_soc_steps/predicted_time_s AS predicted_rate_pp_min
FROM per_curve_stage
ORDER BY absolute_error_pct DESC,source,drone_id,stage;
