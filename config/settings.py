from pathlib import Path


# 0.1 Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "outputs"


# 0.2 Default Analysis Scope
DEFAULT_START_DATE = None
DEFAULT_END_DATE = None
DEFAULT_FAMILIES = None
DEFAULT_RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 0.35


# 0.3 Output Files
UNIVERSE_SUMMARY_FILE = OUTPUT_DIR / "universe_summary.csv"
ACTIVE_UNIVERSE_FILE = OUTPUT_DIR / "active_store_universe.csv"
SAMPLE_COMPARISON_FILE = OUTPUT_DIR / "sample_panel_comparison.csv"
SAMPLE_STORE_LIST_FILE = OUTPUT_DIR / "sample_panel_store_list.csv"
EXTRAPOLATION_COMPARISON_FILE = OUTPUT_DIR / "extrapolation_method_comparison.csv"
WEEKLY_ERROR_FILE = OUTPUT_DIR / "weekly_extrapolation_error.csv"
CATEGORY_ERROR_FILE = OUTPUT_DIR / "category_extrapolation_error.csv"
UNCOVERED_STRATA_FILE = OUTPUT_DIR / "uncovered_strata_summary.csv"
MISSING_RETAILER_FILE = OUTPUT_DIR / "missing_retailer_impact_summary.csv"
PANEL_HEALTH_FILE = OUTPUT_DIR / "panel_health_kpi_summary.csv"
DASHBOARD_SUMMARY_FILE = OUTPUT_DIR / "dashboard_summary.csv"
