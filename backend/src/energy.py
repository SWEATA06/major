"""
Energy and carbon modelling helpers for the cloud auto-scaling simulator.

These functions turn a compute state (instance count + CPU utilisation) into
physical quantities: electrical energy consumed and the resulting carbon
emissions. They are intentionally simple, deterministic, and well-documented so
the numbers can be defended in a report.

Key ideas
---------
1. A server is not "free" when idle: it still draws a large fraction of its
   peak power. We model power as a linear function of CPU utilisation between an
   idle floor and a peak ceiling.
2. Datacenter overhead (cooling, power distribution) is captured with a PUE
   (Power Usage Effectiveness) multiplier applied to IT energy.
3. The electricity grid is not equally "clean" through the day. Carbon
   intensity (gCO2 per kWh) is low around midday (solar) and high in the
   evening peak. carbon_intensity() returns a realistic daily curve.
"""

# --- Physical constants (tunable) ---------------------------------------------

# Per-server power envelope in watts.
P_IDLE_WATTS = 100.0   # power drawn by an idle-but-on server
P_MAX_WATTS = 250.0    # power drawn by a fully utilised server

# Datacenter Power Usage Effectiveness (facility energy / IT energy).
# 1.0 is a perfect datacenter; ~1.5 is a realistic industry average.
PUE = 1.5

# Length of one simulation interval, expressed in hours.
# The simulator advances one step per interval; a 5-minute interval = 1/12 h.
INTERVAL_HOURS = 5.0 / 60.0

# Fallback grid carbon intensity (gCO2/kWh) if no hour is provided.
DEFAULT_CARBON_INTENSITY = 400.0


def instance_power_watts(instances, cpu_utilization):
    """
    Total IT power (watts) for `instances` servers at `cpu_utilization` percent.

    Power per server scales linearly from P_IDLE_WATTS (0% CPU) to
    P_MAX_WATTS (100% CPU). CPU is clamped to [0, 100] for safety.
    """
    cpu = max(0.0, min(float(cpu_utilization), 100.0))
    per_server = P_IDLE_WATTS + (P_MAX_WATTS - P_IDLE_WATTS) * (cpu / 100.0)
    return float(instances) * per_server


def energy_kwh(instances, cpu_utilization, interval_hours=INTERVAL_HOURS, pue=PUE):
    """
    Total facility energy (kWh) consumed during one interval.

    facility_energy = IT_power_watts * interval_hours / 1000 * PUE
    """
    it_watts = instance_power_watts(instances, cpu_utilization)
    it_kwh = it_watts * float(interval_hours) / 1000.0
    return it_kwh * float(pue)


def carbon_intensity(hour):
    """
    Grid carbon intensity (gCO2/kWh) for a given hour of day (0-23).

    Models a realistic daily curve:
      - Overnight (low demand, some baseload): moderate.
      - Midday (solar generation): cleanest.
      - Evening peak (gas/coal peakers): dirtiest.

    Values are illustrative but in a realistic range (~150-550 gCO2/kWh).
    """
    try:
        h = int(hour) % 24
    except (TypeError, ValueError):
        return DEFAULT_CARBON_INTENSITY

    # Hourly curve (gCO2/kWh). Index = hour of day.
    curve = [
        320, 300, 290, 285, 290, 310,   # 00-05 overnight baseload
        360, 400, 380, 300, 240, 190,   # 06-11 morning ramp, solar rising
        160, 150, 160, 190, 250, 340,   # 12-17 solar peak -> ramp down
        480, 540, 520, 460, 400, 350,   # 18-23 evening peak (dirtiest)
    ]
    return float(curve[h])


def carbon_grams(energy_kwh_value, hour):
    """
    Carbon emitted (grams CO2) for a given energy amount at a given hour.
    """
    return float(energy_kwh_value) * carbon_intensity(hour)
