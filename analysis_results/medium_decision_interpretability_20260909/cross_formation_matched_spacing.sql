WITH ranked AS (
  SELECT *, ROW_NUMBER() OVER (
    PARTITION BY charging_pad_availability, wind_direction, wind_level, formation
    ORDER BY total_time_seconds, inter_drone_spacing_cm
  ) AS formation_rank FROM candidate_rankings
)
SELECT s.charging_pad_availability AS charging_pads, s.wind_direction, s.wind_level,
s.formation AS selected_formation,
s.inter_drone_spacing_cm AS selected_spacing_cm,
s.total_time_seconds AS selected_total_time_s,
s.charging_seconds_by_position AS selected_charging_seconds,
s.charging_pad_groups AS selected_pad_groups,
s.charging_pad_loads_seconds AS selected_pad_loads_s,
ps.position_1_discharge_rate_Bideal_pp_min AS selected_p1_rate,
ps.position_2_discharge_rate_Bideal_pp_min AS selected_p2_rate,
ps.position_3_discharge_rate_Bideal_pp_min AS selected_p3_rate,
ps.position_4_discharge_rate_Bideal_pp_min AS selected_p4_rate,
ps.position_5_discharge_rate_Bideal_pp_min AS selected_p5_rate,
(ps.position_1_discharge_rate_Bideal_pp_min + ps.position_2_discharge_rate_Bideal_pp_min + ps.position_3_discharge_rate_Bideal_pp_min + ps.position_4_discharge_rate_Bideal_pp_min + ps.position_5_discharge_rate_Bideal_pp_min) * 25.0 / 60.0 AS selected_swarm_drop_pp,
c.formation AS comparator_formation,
c.inter_drone_spacing_cm AS comparator_spacing_cm,
c.total_time_seconds AS comparator_total_time_s,
c.charging_seconds_by_position AS comparator_charging_seconds,
c.charging_pad_groups AS comparator_pad_groups,
c.charging_pad_loads_seconds AS comparator_pad_loads_s,
pc.position_1_discharge_rate_Bideal_pp_min AS comparator_p1_rate,
pc.position_2_discharge_rate_Bideal_pp_min AS comparator_p2_rate,
pc.position_3_discharge_rate_Bideal_pp_min AS comparator_p3_rate,
pc.position_4_discharge_rate_Bideal_pp_min AS comparator_p4_rate,
pc.position_5_discharge_rate_Bideal_pp_min AS comparator_p5_rate,
(pc.position_1_discharge_rate_Bideal_pp_min + pc.position_2_discharge_rate_Bideal_pp_min + pc.position_3_discharge_rate_Bideal_pp_min + pc.position_4_discharge_rate_Bideal_pp_min + pc.position_5_discharge_rate_Bideal_pp_min) * 25.0 / 60.0 AS comparator_swarm_drop_pp
FROM ranked s JOIN ranked c
 ON s.charging_pad_availability = c.charging_pad_availability
 AND s.wind_direction = c.wind_direction AND s.wind_level = c.wind_level
 AND s.formation != c.formation
JOIN profiles ps ON ps.wind_direction = s.wind_direction AND ps.wind_level = s.wind_level
 AND ps.formation = s.formation AND ps.inter_drone_spacing_cm = s.inter_drone_spacing_cm
JOIN profiles pc ON pc.wind_direction = c.wind_direction AND pc.wind_level = c.wind_level
 AND pc.formation = c.formation AND pc.inter_drone_spacing_cm = c.inter_drone_spacing_cm
WHERE s.rank = 1 AND c.inter_drone_spacing_cm = s.inter_drone_spacing_cm
ORDER BY s.wind_direction, s.wind_level, s.charging_pad_availability, c.total_time_seconds, c.formation;
