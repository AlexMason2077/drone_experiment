"""Mechanism-focused wording; no changes to rates, labels or optimizer scores."""


def interpretation(decision):
    """Separate geometry-based interpretation from the measured charging driver."""
    f = decision["configuration"]["formation"]
    w = decision["input"]["wind_direction"]
    if f == "echalon" and w == "head":
        return ("Staggering can reduce direct downwash exposure by shifting downstream rotors laterally relative to upstream wake cores.", ["R10"])
    if f == "echalon" and w == "side":
        return ("Diagonal placement changes both streamwise and lateral separation, redistributing wake-induced forces and the rotor corrections needed to hold position.", ["R4", "R5"])
    if f == "echalon":
        return ("In tailwind, relative airflow changes upstream–downstream exposure; a staggered layout changes where the convected wakes meet neighbouring rotors.", ["R6"])
    if f == "front" and w == "head":
        return ("A transverse front row avoids a streamwise chain of followers; lateral separation controls neighbouring wake overlap and induced-flow loading.", ["R6"])
    if f == "front" and w == "side":
        return ("When the row lies along the crosswind, upstream sheltering can reduce downstream drag-compensation demand, while rotor wakes modify the local inflow.", ["R7"])
    if f == "column" and w == "side":
        return ("A column transverse to sidewind reduces serial upstream–downstream exposure, while lateral wake interaction remains dependent on spacing.", ["R4"])
    if f == "column":
        return ("A wind-aligned column couples shielding and wake-induced thrust changes along the line; these mechanisms can create unequal position loads rather than uniform savings.", ["R4"])
    if f == "vee":
        return ("Under tailwind, the two arms of the V have different upstream–downstream exposure from the headwind case, changing shielding and wake interception. "
                "Here P2's nearly flat integer-SOC record also strongly lowers its calculated charging time; it should not be interpreted as an energy-free position.", ["R6"])
    return ("Diamond geometry distributes positions both along and across the airflow, combining sheltered locations with laterally offset rotor inflow.", ["R7", "R10"])


def mechanism_sections(citations):
    """Mechanisms are explained once in depth, with short application per result."""
    return [
        ("mechanism_load", "Why wind changes a single drone's power demand",
         "For approximately steady, level flight, the rotor system supplies vertical thrust to support weight and horizontal thrust to balance aerodynamic drag. "
         "Ignoring vertical airframe forces, T cos(theta) = mg and T sin(theta) = D, so T = sqrt((mg)^2 + D^2). "
         "Greater drag therefore increases the thrust required for force balance. A changed rotor inflow also changes induced power, "
         "so total electrical power reflects both effects rather than drag alone. " + citations["R1"] + " " + citations["R2"]),
        ("mechanism_wake", "Wake location changes the thrust produced for the same motor effort",
         "The relevant distinction is the descending wake core versus the flow around its edges. "
         "In rotor-rig experiments, tandem downwash reduced downstream thrust, whereas suitable oblique placement improved loading through wake-edge upwash. "
         "Maintaining the same required thrust then involves a different rotor speed and power. This explains why lateral staggering can matter, "
         "rather than merely the distance between two centres. " + citations["R10"]),
        ("mechanism_control", "Position holding converts uneven aerodynamic loads into motor corrections",
         "A disturbed rotor does not operate independently of the flight controller. Uneven forces and moments require differential motor commands to maintain attitude and position. "
         "Single-quadcopter flight measurements show power fluctuations during motor-speed corrections, and close-proximity control experiments explicitly model neighbour-induced disturbances. "
         "Consequently, the empirical discharge rate includes propulsion and the response to disturbance together. " + citations["R2"] + " " + citations["R5"]),
        ("mechanism_direction", "Tailwind changes relative airflow, not just the sign of an energy penalty",
         "The aircraft's velocity relative to the air is v_air = v_ground − v_wind. At the fixed 0.1 m/s ground speed, "
         "a following wind faster than the drone reverses the longitudinal relative airflow and requires a balancing force to avoid being carried ahead. "
         "This changes the order in which neighbours intercept wind-convected wakes. Therefore headwind and tailwind cannot be assigned the same leader–follower explanation "
         "or treated as equal and opposite power corrections. Wake structure and single-vehicle power vary with the flow regime. " + citations["R2"] + " " + citations["R6"]),
        ("mechanism_spacing", "Spacing changes both beneficial inflow and adverse interference",
         "Moving from 50 to 75 cm changes streamwise distance, lateral offset and the part of a neighbour's flow sampled by a rotor. "
         "It can reduce direct downwash exposure, but can also reduce sheltering or move a rotor away from a favourable inflow region. "
         "The outcome depends on geometry and wind direction, not spacing alone. Formation simulations likewise show that adding rotor-induced forces can change a drag-only ranking. "
         + citations["R4"] + " " + citations["R7"]),
    ]
