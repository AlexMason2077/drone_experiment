"""Effective battery identity for a documented wind-tunnel metadata mistake.

The original CSV files remain untouched.  Only side-wind wind-tunnel records
for drone 5 are interpreted as B12; other B06 observations stay B06.
"""


def corrected_battery_id(row):
    protocol = str(row.get("protocol") or row.get("soc_mode") or "").strip().lower()
    wind_direction = str(row.get("wind_direction") or "").strip().lower()
    drone_name = str(row.get("drone_name") or "").strip().lower()
    drone_number = str(row.get("drone_number") or "").strip()
    if protocol == "wind_tunnel" and wind_direction == "side wind" and (
        drone_name == "drone_5" or drone_number == "5"
    ):
        return "B12"
    return row.get("battery_id", "")


def correct_battery_row(row):
    """Return a corrected copy, never mutate the source row."""
    result = dict(row)
    result["battery_id"] = corrected_battery_id(result)
    return result


def correct_battery_dataframe(frame):
    """Apply the same rule to an in-memory pandas frame without writing CSV."""
    needed = {"battery_id", "wind_direction", "drone_name"}
    if frame.empty or not needed.issubset(frame.columns):
        return frame
    protocol_column = "protocol" if "protocol" in frame else "soc_mode"
    if protocol_column not in frame:
        return frame
    mask = (
        frame[protocol_column].astype(str).str.strip().str.lower().eq("wind_tunnel")
        & frame["wind_direction"].astype(str).str.strip().str.lower().eq("side wind")
        & frame["drone_name"].astype(str).str.strip().str.lower().eq("drone_5")
    )
    if not mask.any():
        return frame
    corrected = frame.copy()
    corrected.loc[mask, "battery_id"] = "B12"
    return corrected
