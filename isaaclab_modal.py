"""Run Isaac Lab RL training headless on a Modal GPU.

Usage (inside the venv):
    modal run isaaclab_modal.py                                   # quick Cartpole smoke test
    modal run isaaclab_modal.py --task Isaac-Ant-v0 --max-iterations 500
    modal volume get isaaclab-logs / ./logs                       # download checkpoints/logs
"""

import subprocess

import modal

ISAACLAB_DIR = "/workspace/isaaclab"

image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-lab:2.3.2", add_python="3.11")
    .entrypoint([])  # image defaults to /isaac-sim/runheadless.sh
    .env(
        {
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "OMNI_KIT_ACCEPT_EULA": "YES",
        }
    )
)

logs = modal.Volume.from_name("isaaclab-logs", create_if_missing=True)

app = modal.App("isaaclab", image=image)


@app.function(gpu="L40S", timeout=4 * 60 * 60, volumes={f"{ISAACLAB_DIR}/logs": logs})
def train(task: str, num_envs: int, max_iterations: int):
    cmd = [
        f"{ISAACLAB_DIR}/isaaclab.sh", "-p",
        "scripts/reinforcement_learning/rsl_rl/train.py",
        "--task", task,
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--headless",
    ]
    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=ISAACLAB_DIR, check=True)
    finally:
        logs.commit()


@app.local_entrypoint()
def main(task: str = "Isaac-Cartpole-v0", num_envs: int = 1024, max_iterations: int = 50):
    train.remote(task, num_envs, max_iterations)
