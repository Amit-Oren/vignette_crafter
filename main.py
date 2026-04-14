"""Entry point — loads a simulation config YAML and runs the experiment."""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml

from agents.base_agent import set_experiment_dir
from configs.logging_config import setup_logging
from simulation.runner import SimulationRunner


def main():
    parser = argparse.ArgumentParser(description="Run a vignette simulation.")
    parser.add_argument(
        "--config", "-c",
        default="configs/simulation_config.yaml",
        help="Path to the simulation YAML config (default: configs/simulation_config.yaml)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim  = cfg["simulation"]
    models = cfg["models"]

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline     = sim.get("pipeline", "vignette")
    experiment_dir = Path("data/output") / f"{pipeline}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(experiment_dir)
    set_experiment_dir(experiment_dir)

    runner = SimulationRunner(
        num_patients    = sim["num_patients"],
        patient_ids     = sim.get("patient_ids"),
        seed            = sim["seed"],
        max_retries     = sim["max_retries"],
        temperature     = sim["temperature"],
        pipeline        = pipeline,
        persona_context  = sim["persona_context"],
        use_formulation  = sim["use_formulation"],
        n_items          = sim["self_report_items"],
        models          = models,
        experiment_dir  = experiment_dir,
    )
    runner.run()


if __name__ == "__main__":
    main()
