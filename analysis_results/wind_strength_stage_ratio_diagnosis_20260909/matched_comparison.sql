SELECT a.wind_direction,a.formation,a.spacing_cm,a.battery_id,a.drone_id,
 a.source AS source_lv1,b.source AS source_lv2,
 a.experiment_id AS experiment_lv1,b.experiment_id AS experiment_lv2,
 a.run_id AS run_lv1,b.run_id AS run_lv2,
 a.high_k/a.medium_k AS high_ratio_lv1,b.high_k/b.medium_k AS high_ratio_lv2,
 a.low_k/a.medium_k AS low_ratio_lv1,b.low_k/b.medium_k AS low_ratio_lv2,
 100.0*((b.high_k/b.medium_k)/(a.high_k/a.medium_k)-1.0) AS high_change_pct,
 100.0*((b.low_k/b.medium_k)/(a.low_k/a.medium_k)-1.0) AS low_change_pct,
 100.0*(b.high_raw_rate_pp_min/a.high_raw_rate_pp_min-1.0) AS high_rate_change_pct,
 100.0*(b.medium_raw_rate_pp_min/a.medium_raw_rate_pp_min-1.0) AS medium_rate_change_pct,
 100.0*(b.low_raw_rate_pp_min/a.low_raw_rate_pp_min-1.0) AS low_rate_change_pct,
 a.high_starts_after_all_ready AND b.high_starts_after_all_ready AS both_high_after_ready,
 a.low_ends_before_first_landing AND b.low_ends_before_first_landing AS both_full_low_before_landing,
 a.low_after_first_landing_fraction AS low_after_landing_fraction_lv1,
 b.low_after_first_landing_fraction AS low_after_landing_fraction_lv2
FROM full_curves a JOIN full_curves b
 ON a.wind_direction=b.wind_direction AND a.formation=b.formation
 AND a.spacing_cm=b.spacing_cm AND a.battery_id=b.battery_id AND a.drone_id=b.drone_id
WHERE a.wind_level=1 AND b.wind_level=2
ORDER BY a.wind_direction,a.formation,a.spacing_cm,a.battery_id;
