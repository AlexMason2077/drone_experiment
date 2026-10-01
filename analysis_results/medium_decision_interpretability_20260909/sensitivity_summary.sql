-- Whole-trial resampling is produced by interpret_medium_decisions.py.
-- Reconcile leave-one-out denominators directly from the deletion-level audit.
WITH deletion_counts AS (
  SELECT key,
         SUM(CASE WHEN evaluable = 1 THEN 1 ELSE 0 END) AS valid_deletions,
         SUM(CASE WHEN evaluable = 1 AND retained = 1 THEN 1 ELSE 0 END) AS retained_deletions
  FROM leave_one_trial_out
  GROUP BY key
)
SELECT s.key, s.bootstrap_attempts, s.bootstrap_valid, s.bootstrap_invalid,
       s.nominal_winner_selection_frequency,
       d.valid_deletions, d.retained_deletions,
       1.0 * d.retained_deletions / NULLIF(d.valid_deletions, 0) AS deletion_retention_fraction
FROM resampling_sensitivity AS s
JOIN deletion_counts AS d ON s.key = d.key
ORDER BY s.key;
