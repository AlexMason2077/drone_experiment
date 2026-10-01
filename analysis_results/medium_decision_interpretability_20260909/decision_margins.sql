-- Source: all 275 frozen candidate scores; no simulated flights.
-- The original charging calculation is separately documented and audited.
WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY charging_pad_availability, wind_direction, wind_level
    ORDER BY total_time_seconds, formation, inter_drone_spacing_cm
  ) AS computed_rank
  FROM candidate_rankings
)
SELECT b.charging_pad_availability AS charging_pads,
       b.wind_direction,
       CASE b.wind_level WHEN 1 THEN 'low' ELSE 'high' END AS wind_strength,
       CASE b.formation WHEN 'echalon' THEN 'echelon' ELSE b.formation END
           || ' ' || b.inter_drone_spacing_cm || ' cm' AS configuration,
       b.total_time_seconds / 60.0 AS total_time_min,
       r.total_time_seconds - b.total_time_seconds AS gap_s,
       CASE r.formation WHEN 'echalon' THEN 'echelon' ELSE r.formation END
           || ' ' || r.inter_drone_spacing_cm || ' cm' AS runner_up,
       b.min_position_run_count AS min_trials
FROM ranked AS b
JOIN ranked AS r
  ON b.charging_pad_availability = r.charging_pad_availability
 AND b.wind_direction = r.wind_direction
 AND b.wind_level = r.wind_level
 AND r.computed_rank = 2
WHERE b.computed_rank = 1
ORDER BY b.wind_direction, b.wind_level, b.charging_pad_availability;
