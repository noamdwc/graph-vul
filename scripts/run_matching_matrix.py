"""Run the shared-seed matrix or a reduced real-model integration pilot."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    experiments = ["pair", "graph_pair", "triplet_random", "triplet_hard"]
    seeds = [42] if args.pilot else [42, 43, 44]
    manifests = []
    data_root = args.data_root
    if args.pilot:
        from graph_vul.pipeline.prepare_matching import prepare_pilot

        data_root = prepare_pilot(data_root, args.output_dir / "prepared")
    for experiment in experiments:
        for seed in seeds:
            output = args.output_dir / f"{experiment}-{seed}"
            command = [
                sys.executable,
                "-m",
                "graph_vul.train_finetune",
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output),
                "--ablation-mode",
                experiment,
                "--seed",
                str(seed),
            ]
            if args.pilot:
                command += [
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--grad-accum",
                    "1",
                ]
            subprocess.run(command, check=True)
            external_command = [
                sys.executable,
                "-m",
                "graph_vul.eval_external",
                "--checkpoint",
                str(output / "model.ckpt"),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output / "external"),
            ]
            if args.pilot:
                external_command += ["--max-samples", "32"]
            subprocess.run(external_command, check=True)
            manifests.extend(
                [
                    output / "predictions/manifest.json",
                    output / "external/manifest.json",
                ]
            )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "graph_vul.analysis.cwe_matching_report",
            "--manifests",
            *map(str, manifests),
            "--output-dir",
            str(args.output_dir / "report"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
