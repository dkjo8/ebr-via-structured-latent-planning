"""
Modal deployment for EBRM paper figure generation on a T4 GPU.

Usage:
    modal run modal_run.py                    # Run figure generation
    modal run modal_run.py::run_sweeps        # Run hyperparameter sweeps
    modal run modal_run.py::run_all           # Run full pipeline

    modal volume get ebrm-results / ./modal_output/   # Download results
"""

import modal

app = modal.App("ebrm-gpu")

vol = modal.Volume.from_name("ebrm-results", create_if_missing=True)

JULIA_VERSION = "1.10.8"
JULIA_URL = f"https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-{JULIA_VERSION}-linux-x86_64.tar.gz"
JULIA_BIN = f"/opt/julia-{JULIA_VERSION}/bin"
JULIA_PATH_SETUP = f'export PATH="{JULIA_BIN}:$PATH"'

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("curl", "ca-certificates", "git", "wget")
    .run_commands(
        f"wget -q {JULIA_URL} -O /tmp/julia.tar.gz && "
        f"tar -xzf /tmp/julia.tar.gz -C /opt && "
        f"rm /tmp/julia.tar.gz && "
        f"ln -s /opt/julia-{JULIA_VERSION}/bin/julia /usr/local/bin/julia && "
        f"julia --version",
    )
    .run_commands(
        'julia -e "'
        "using CUDA; "
        'CUDA.set_runtime_version!(local_toolkit=true)"',
    )
    .add_local_dir(
        ".",
        remote_path="/app",
        copy=True,
        ignore=[
            ".github",
            ".DS_Store",
            "runs",
            "Manifest.toml",
            ".git",
            "notebooks",
            "modal_output",
            "__pycache__",
            "*.pyc",
        ],
    )
    .run_commands(
        'cd /app && julia --project=. -e "'
        "using Pkg; "
        "Pkg.instantiate(); "
        "Pkg.precompile()"
        '"',
        gpu="T4",
    )
)

RESULTS_DIR = "/results"


@app.function(
    gpu="T4",
    image=image,
    volumes={RESULTS_DIR: vol},
    timeout=7200,
)
def generate_figures():
    """Train all 3 tasks and generate every paper figure on a T4 GPU."""
    import subprocess

    result = subprocess.run(
        ["julia", "--project=/app", "/app/analysis/generate_paper_figures.jl"],
        capture_output=False,
        text=True,
        cwd="/app",
    )

    import shutil
    import os

    figures_src = "/app/analysis/figures"
    runs_src = "/app/runs"

    if os.path.isdir(figures_src):
        shutil.copytree(figures_src, f"{RESULTS_DIR}/figures", dirs_exist_ok=True)
        print(f"Copied figures to {RESULTS_DIR}/figures")

    if os.path.isdir(runs_src):
        shutil.copytree(runs_src, f"{RESULTS_DIR}/runs", dirs_exist_ok=True)
        print(f"Copied runs to {RESULTS_DIR}/runs")

    vol.commit()

    figure_count = len(os.listdir(f"{RESULTS_DIR}/figures")) if os.path.isdir(f"{RESULTS_DIR}/figures") else 0
    print(f"\nDone. {figure_count} figures saved to volume 'ebrm-results'.")
    print("Download with: modal volume get ebrm-results / ./modal_output/")

    return result.returncode


@app.function(
    gpu="T4",
    image=image,
    volumes={RESULTS_DIR: vol},
    timeout=14400,
)
def run_sweeps():
    """Run hyperparameter sweeps for all tasks on a T4 GPU."""
    import subprocess
    import shutil
    import os

    result = subprocess.run(
        ["julia", "--project=/app", "/app/experiments/run_sweep.jl", "all"],
        capture_output=False,
        text=True,
        cwd="/app",
    )

    analysis_src = "/app/analysis"
    for f in os.listdir(analysis_src):
        if f.startswith("sweep_results_") and f.endswith(".csv"):
            shutil.copy2(os.path.join(analysis_src, f), f"{RESULTS_DIR}/{f}")

    vol.commit()
    print("\nSweep results saved to volume 'ebrm-results'.")
    return result.returncode


@app.function(
    gpu="T4",
    image=image,
    volumes={RESULTS_DIR: vol},
    timeout=21600,
)
def run_all():
    """Run the full experimental pipeline on a T4 GPU."""
    import subprocess
    import shutil
    import os

    result = subprocess.run(
        ["julia", "--project=/app", "/app/experiments/run_all.jl"],
        capture_output=False,
        text=True,
        cwd="/app",
    )

    for src_dir in ["/app/analysis/figures", "/app/analysis", "/app/runs"]:
        if os.path.isdir(src_dir):
            dest = f"{RESULTS_DIR}/{os.path.basename(src_dir)}"
            shutil.copytree(src_dir, dest, dirs_exist_ok=True)

    vol.commit()
    print("\nFull pipeline results saved to volume 'ebrm-results'.")
    return result.returncode


@app.local_entrypoint()
def main():
    """Default: generate paper figures."""
    exit_code = generate_figures.remote()
    if exit_code != 0:
        print(f"Julia exited with code {exit_code}")
    else:
        print("\nSuccess! Download results with:")
        print("  modal volume get ebrm-results / ./modal_output/")
