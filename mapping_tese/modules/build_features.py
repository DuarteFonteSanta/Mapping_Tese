"""CLI entry point to (re)build the per-subject feature cache."""
from mapping_tese.modules import constants as C
from mapping_tese.modules.cache import build_all_subjects_cache

SCHEME = "s1_hrf6_win1"
CACHE_DIR = C.PROJECT_ROOT / "data" / "extracted" / SCHEME

if __name__ == "__main__":
    build_all_subjects_cache(CACHE_DIR)