SELECT a.wind_direction,a.formation,a.spacing_cm,a.battery_id,a.drone_id,a.stage,
      a.source AS source_lv1,b.source AS source_lv2,a.ratio AS ratio_lv1,b.ratio AS ratio_lv2,
      100.0*(b.ratio/a.ratio-1.0) AS change_pct
      FROM expanded_stage_pairs a JOIN expanded_stage_pairs b
      ON a.wind_direction=b.wind_direction AND a.formation=b.formation AND a.spacing_cm=b.spacing_cm
      AND a.battery_id=b.battery_id AND a.drone_id=b.drone_id AND a.stage=b.stage
      WHERE a.wind_level=1 AND b.wind_level=2
      ORDER BY a.stage,a.wind_direction,a.formation,a.battery_id;
