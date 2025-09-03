import subprocess
import sys

# Ordered list of your scripts (stages 02–06)
scripts = [
    # 02
    "02_compute_station_bias_factors_zeta_from_pairs_XAVIER.py",
    "02_compute_station_bias_factors_zeta_from_pairs_PERSIANN.py",
    "02_compute_station_bias_factors_zeta_from_pairs_IMERG.py",
    "02_compute_station_bias_factors_zeta_from_pairs.py",

    # 03
    "03_interpolate_zeta_to_grid_IDW_or_kriging_XAVIER.py",
    "03_interpolate_zeta_to_grid_IDW_or_kriging_PERSIANN.py",
    "03_interpolate_zeta_to_grid_IDW_or_kriging_IMERG.py",
    "03_interpolate_zeta_to_grid_IDW_or_kriging.py",

    # 04
    "04_apply_bias_correction_and_density_plots_XAVIER.py",
    "04_apply_bias_correction_and_density_plots_PERSIANN.py",
    "04_apply_bias_correction_and_density_plots_IMERG.py",
    "04_apply_bias_correction_and_density_plots_CHRIPS.py",

    # 05
    "05_apply_bias_to_AMaxDaily_XAVIER.py",
    "05_apply_bias_to_AMaxDaily_PERSIANN.py",
    "05_apply_bias_to_AMaxDaily_IMERG.py",
    "05_apply_bias_to_AMaxDaily_CHIRPS.py",

    # Final one (must be last!)
    "06_apply_bias_correction_and_density_plots_ALL_4x2.py"
]

# Run them in order
for script in scripts:
    print(f"\n[RUNNING] {script}")
    try:
        subprocess.run([sys.executable, script], check=True)
        print(f"[DONE] {script}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script} failed with code {e.returncode}")
        sys.exit(e.returncode)

print("\n✅ Pipeline finished successfully.")
