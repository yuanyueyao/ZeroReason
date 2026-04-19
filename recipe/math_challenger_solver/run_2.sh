#!/usr/bin/env bash
# Second stage after run.sh (invoked by run_sequential.sh). Replace with your command(s).
set -euo pipefail
# Example: same entrypoint with different experiment_name / checkpoint:
# python3 -m recipe.math_challenger_solver.main_ppo trainer.experiment_name=demo2 "$@"
