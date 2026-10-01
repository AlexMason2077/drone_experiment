"""GUI-only preview of the Wind Tunnel controller's existing target positions.

The collector never imports this module. Changing a wind arrow here cannot
change a flight target or an RC command.
"""
import math


def layout_description(formation, wind_direction, spacing_cm):
    formation = str(formation).strip().lower()
    if formation in {"echalon", "echolon", "echlon"}:
        formation = "echelon"
    wind = str(wind_direction).strip().lower()
    if formation not in {"front", "vee", "column", "diamond", "echelon"}:
        raise ValueError(f"Unknown Wind Tunnel formation: {formation}")
    if wind not in {"head wind", "tail wind", "side wind"}:
        raise ValueError(f"Unknown Wind Tunnel direction: {wind}")
    spacing = float(spacing_cm)
    if spacing not in (50, 75):
        raise ValueError("Wind Tunnel spacing must be 50 or 75 cm.")

    arm = 75.0 / math.sqrt(2.0)
    side = wind == "side wind"
    if formation == "front":
        positions = [(0.0, i * spacing) for i in range(5)]
    elif formation == "vee" and spacing == 75:
        positions = ([(0, 0), (arm, arm), (2*arm, 2*arm), (arm, 3*arm), (0, 4*arm)]
                     if side else
                     [(0, 0), (arm, arm), (2*arm, 2*arm), (3*arm, arm), (4*arm, 0)])
    elif formation == "vee":
        step = 25.0 * math.sqrt(2.0)
        positions = [(0, 0), (step, step), (2*step, 2*step),
                     (3*step, step), (4*step, 0)]
    elif formation == "echelon" and spacing == 75:
        positions = ([(4*arm, 0), (3*arm, arm), (2*arm, 2*arm),
                      (arm, 3*arm), (0, 4*arm)] if side else
                     [(0, 4*arm), (arm, 3*arm), (2*arm, 2*arm),
                      (3*arm, arm), (4*arm, 0)])
    elif formation == "echelon":
        root2 = math.sqrt(2.0)
        positions = [(0, 150*root2), (35*root2, 115*root2),
                     (75*root2, 75*root2), (115*root2, 35*root2),
                     (150*root2, 0)]
    elif formation == "column" and spacing == 75:
        positions = ([(i*75, 0) for i in range(5)] if side else
                     [(0, (4-i)*75) for i in range(5)])
    elif formation == "column" and wind in {"head wind", "tail wind", "side wind"} and spacing == 50:
        positions = [(0, (4-i)*spacing) for i in range(5)]
    elif formation == "column":
        positions = [(i*50, 0) for i in range(5)]
    elif formation == "diamond" and spacing == 75:
        positions = ([(0,75),(75,0),(75,75),(75,150),(150,75)] if side else
                     [(75,0),(0,75),(75,75),(150,75),(75,150)])
    elif formation == "diamond":
        positions = [(50,0),(0,50),(50,50),(100,50),(50,100)]
    else:
        positions = [(i*50, 0) for i in range(5)]

    # These orientation flags mirror the committed collector configuration.
    pad_x_aligned_with_body_forward = formation == "front" or (
        side and spacing == 75 and formation in {"vee", "echelon", "column", "diamond"}
    )
    nose = "+X" if pad_x_aligned_with_body_forward else "+Y"
    if side and formation in {"echelon", "vee", "column", "diamond"} and spacing == 50:
        source, vector = "+X", [-1, 0]
    elif side:
        source, vector = "+Y", [0, -1]
    elif wind == "head wind":
        source, vector = nose, ([-1, 0] if nose == "+X" else [0, -1])
    else:
        source, vector = nose.replace("+", "-"), ([1, 0] if nose == "+X" else [0, 1])
    destination = ("-" if source[0] == "+" else "+") + source[1]
    note = ""
    if formation == "echelon" and spacing == 50:
        note = ("The GitHub controller's fixed target separations are 70, 80, 80, "
                "and 70 cm, despite the 50 cm setting.")
    elif formation == "column" and wind in {"head wind", "tail wind", "side wind"} and spacing == 50:
        note = ("Pad 5 is at the +Y/front end, followed by 6, 7, 8, 1 toward -Y; "
                "adjacent pad centres are 50 cm apart.")
    return {
        "pad_ids": [5, 6, 7, 8, 1],
        "positions": [[x, y] for x, y in positions],
        "pad_axes": [[1, 0], [0, 1]],
        "pad_arrow_global": "+X",
        "nose": nose,
        "screen_axes": [[1, 0], [0, -1]],
        "flow": {"source": source, "vector": vector,
                 "text": f"{source} -> {destination}"},
        "spacing_definition": "controller target positions",
        "pad_x_aligned_with_body_forward": pad_x_aligned_with_body_forward,
        "heading_tolerance_required": pad_x_aligned_with_body_forward,
        "layout_note": note,
    }
