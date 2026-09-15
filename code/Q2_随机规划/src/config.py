from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
OUTPUT = ROOT / 'outputs_v2'

ATTACHMENT1 = DATA / 'attachment1.xlsx'
ATTACHMENT2 = DATA / 'attachment2.xlsx'
RESULT2_TEMPLATE = DATA / 'result2_template.xlsx'

DT_HOURS = 1/6
SOC_MIN_KWH = 1200.0
SOC_MAX_KWH = 10800.0
SOC_INITIAL_KWH = 6000.0
SOC_FINAL_KWH = 6000.0
P_CHARGE_MAX_KW = 5000.0
P_DISCHARGE_MAX_KW = 5000.0
ETA_C = 0.90
ETA_D = 0.90
EMERGENCY_MULTIPLIER = 5.0

# Q2-v2: rolling recent history + conformal calibration + correlated residual scenarios
TRAIN_WINDOW_DAYS = 120
CALIBRATION_DAYS = 14
RECENCY_HALF_LIFE_DAYS = 45.0
N_SCENARIOS = 5
TARGET_COVERAGE = 0.82
RANDOM_SEED = 20260911
REPRESENTATIVE_DATES = ['2025-03-20','2025-06-21','2025-09-23','2025-12-21']

# V4 adaptive conformal
ADAPTIVE_CALIBRATION_DAYS = 21
ADAPTIVE_TIME_BLOCKS = 6  # 6 blocks × 4 hours
