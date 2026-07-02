"""Command line interface for SkillRQ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .agent_sim import evaluate_tool_call_plans, run_vllm_tool_call_planning, simulate_mock_tool_calls
from .capability import build_capability_processed_data
from .codebook import build_m3_codebooks
from .codebook.runner import DEFAULT_M3_DATASETS
from .config.loader import load_paths_config
from .data import build_skillret_processed_data
from .diagnostics import run_diagnostics
from .m4 import prepare_m4_data
from .m4.baselines import train_rq_kmeans, train_rq_vae
from .m4.evaluate import evaluate_m4_predictions
from .m4.predict import predict_query_codes
from .m4.sequence_split import build_sequence_eval_view
from .m4.soft import predict_soft_multipath_codes, train_soft_multipath_code_predictor
from .m4.train import train_capabilityrq
from .m5 import prepare_m5_data
from .m5.evaluate import evaluate_m5_predictions
from .m5.planning import prepare_code_path_planning_data, predict_code_path_plan, train_code_path_planner
from .m5.predict import predict_residual_paths
from .m5.train import train_residual_selector
from .m7 import prepare_m7_data
from .m7.evaluate import evaluate_reranked_predictions
from .m7.joint_predict import predict_joint_reranked_capabilities
from .m7.joint_train import train_joint_reranker
from .m7.predict import predict_reranked_capabilities
from .m7.train import train_reranker
from .prompting import build_code_guided_prompts
from .retrieval import run_m2_baselines
from .retrieval.runner import DEFAULT_DATASETS, DEFAULT_METHODS, DEFAULT_TOP_K


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skillrq",
        description="SkillRQ experiment toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser("config", help="Inspect configuration.")
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    show_parser = config_subparsers.add_parser("show", help="Show resolved path configuration.")
    show_parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to a flat YAML path configuration file.",
    )

    data_parser = subparsers.add_parser("data", help="Build and inspect processed datasets.")
    data_subparsers = data_parser.add_subparsers(dest="data_command")

    build_data_parser = data_subparsers.add_parser("build", help="Build canonical processed data.")
    build_data_parser.add_argument(
        "--dataset",
        choices=["skillret"],
        default="skillret",
        help="Dataset to normalize. M1 supports SkillRet.",
    )
    build_data_parser.add_argument(
        "--paths",
        type=Path,
        default=None,
        help="Path to a flat YAML path configuration file.",
    )

    capability_parser = subparsers.add_parser("capability", help="Build capability recommendation data.")
    capability_subparsers = capability_parser.add_subparsers(dest="capability_command")
    capability_build_parser = capability_subparsers.add_parser(
        "build",
        help="Build canonical Tool/API capability recommendation data.",
    )
    capability_build_parser.add_argument(
        "--dataset",
        choices=["toolbench", "api_bank", "all"],
        default="all",
        help="Capability dataset to normalize.",
    )
    capability_build_parser.add_argument(
        "--paths",
        type=Path,
        default=None,
        help="Path to a flat YAML path configuration file.",
    )
    capability_build_parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output directory. Defaults to capability_processed_root.",
    )
    capability_build_parser.add_argument(
        "--skip-answer-trees",
        action="store_true",
        help="Skip ToolBench answer tree trajectory extraction.",
    )
    capability_build_parser.add_argument(
        "--limit-tools",
        type=int,
        default=None,
        help="Limit ToolBench tool schema records for smoke tests.",
    )
    capability_build_parser.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        help="Limit ToolBench query records for smoke tests.",
    )

    m2_parser = subparsers.add_parser("m2", help="Run M2 retrieval baselines.")
    m2_subparsers = m2_parser.add_subparsers(dest="m2_command")
    m2_run_parser = m2_subparsers.add_parser("run", help="Run BM25 and dense retrieval baselines.")
    m2_run_parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DEFAULT_DATASETS),
        default=list(DEFAULT_DATASETS),
        help="Datasets to evaluate.",
    )
    m2_run_parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(DEFAULT_METHODS),
        default=list(DEFAULT_METHODS),
        help="Retrieval methods to run.",
    )
    m2_run_parser.add_argument(
        "--top-k",
        default=",".join(str(k) for k in DEFAULT_TOP_K),
        help="Comma-separated K values, e.g. 1,5,10,20.",
    )
    m2_run_parser.add_argument(
        "--max-queries",
        type=int,
        default=300,
        help="Maximum evaluated queries per dataset. Use 0 for all queries.",
    )
    m2_run_parser.add_argument(
        "--max-candidates",
        type=int,
        default=10000,
        help="Maximum candidates per dataset, with gold candidates always retained. Use 0 for all candidates.",
    )
    m2_run_parser.add_argument(
        "--paths",
        type=Path,
        default=None,
        help="Path to a flat YAML path configuration file.",
    )
    m2_run_parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Override baseline artifact directory.",
    )

    m3_parser = subparsers.add_parser("m3", help="Build CapabilityRQ codebook assignments.")
    m3_subparsers = m3_parser.add_subparsers(dest="m3_command")
    m3_build_parser = m3_subparsers.add_parser("build", help="Build M3 semantic code assignments.")
    m3_build_parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DEFAULT_M3_DATASETS),
        default=list(DEFAULT_M3_DATASETS),
        help="Datasets to assign semantic code paths.",
    )
    m3_build_parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Limit assignments per dataset for smoke tests.",
    )
    m3_build_parser.add_argument(
        "--paths",
        type=Path,
        default=None,
        help="Path to a flat YAML path configuration file.",
    )

    m4_parser = subparsers.add_parser("m4", help="Train and run query-to-code models.")
    m4_subparsers = m4_parser.add_subparsers(dest="m4_command")

    m4_prepare_parser = m4_subparsers.add_parser("prepare", help="Prepare M4 supervised query-code data.")
    m4_prepare_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m4_prepare_parser.add_argument("--datasets", nargs="+", default=None)
    m4_prepare_parser.add_argument("--output-root", type=Path, default=None)
    m4_prepare_parser.add_argument("--limit-queries", type=int, default=None)
    m4_prepare_parser.add_argument("--paths", type=Path, default=None)

    m4_train_parser = m4_subparsers.add_parser("train", help="Train CapabilityRQ query-to-code model.")
    m4_train_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m4_train_parser.add_argument("--model-kind", choices=["hard", "soft-multipath"], default="hard")
    m4_train_parser.add_argument("--data-root", type=Path, default=None)
    m4_train_parser.add_argument("--output-root", type=Path, default=None)
    m4_train_parser.add_argument("--epochs", type=int, default=5)
    m4_train_parser.add_argument("--batch-size", type=int, default=512)
    m4_train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    m4_train_parser.add_argument("--embedding-dim", type=int, default=256)
    m4_train_parser.add_argument("--hidden-dim", type=int, default=512)
    m4_train_parser.add_argument("--code-embedding-dim", type=int, default=128)
    m4_train_parser.add_argument("--max-vocab-size", type=int, default=200000)
    m4_train_parser.add_argument("--contrastive-weight", type=float, default=1.0)
    m4_train_parser.add_argument("--hierarchy-weight", type=float, default=1.0)
    m4_train_parser.add_argument("--path-bce-weight", type=float, default=0.2)
    m4_train_parser.add_argument("--contrastive-negative-count", type=int, default=256)
    m4_train_parser.add_argument("--temperature", type=float, default=0.07)
    m4_train_parser.add_argument("--device", default=None)
    m4_train_parser.add_argument("--swanlab-project", default="SkillRQ-M4")
    m4_train_parser.add_argument("--swanlab-run-name", default=None)
    m4_train_parser.add_argument("--disable-swanlab", action="store_true")
    m4_train_parser.add_argument("--paths", type=Path, default=None)

    m4_predict_parser = m4_subparsers.add_parser("predict", help="Predict code paths and retrieve candidates.")
    m4_predict_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m4_predict_parser.add_argument("--model-kind", choices=["hard", "soft-multipath"], default="hard")
    m4_predict_parser.add_argument("--data-root", type=Path, default=None)
    m4_predict_parser.add_argument("--checkpoint-root", type=Path, required=True)
    m4_predict_parser.add_argument("--output-root", type=Path, default=None)
    m4_predict_parser.add_argument("--top-n-paths", type=int, default=8)
    m4_predict_parser.add_argument("--candidate-budget", type=int, default=100)
    m4_predict_parser.add_argument("--beam-width", type=int, default=8)
    m4_predict_parser.add_argument("--score-blend", type=float, default=0.65)
    m4_predict_parser.add_argument("--split", default=None)
    m4_predict_parser.add_argument("--device", default=None)
    m4_predict_parser.add_argument("--swanlab-project", default="SkillRQ-M4")
    m4_predict_parser.add_argument("--swanlab-run-name", default=None)
    m4_predict_parser.add_argument("--disable-swanlab", action="store_true")
    m4_predict_parser.add_argument("--paths", type=Path, default=None)

    m4_eval_parser = m4_subparsers.add_parser("evaluate", help="Evaluate M4 predictions.")
    m4_eval_parser.add_argument("--prediction-path", type=Path, required=True)
    m4_eval_parser.add_argument("--output-path", type=Path, required=True)
    m4_eval_parser.add_argument("--top-k", default="1,5,10,20,50,100")
    m4_eval_parser.add_argument("--set-metric-name", default="tool_set_recall")

    m4_sequence_parser = m4_subparsers.add_parser("sequence-split", help="Build sequence-dev/test view from M4 data.")
    m4_sequence_parser.add_argument("--target", choices=["capability"], default="capability")
    m4_sequence_parser.add_argument("--data-root", type=Path, default=None)
    m4_sequence_parser.add_argument("--output-root", type=Path, default=None)
    m4_sequence_parser.add_argument("--sequence-dev-size", type=int, default=2000)
    m4_sequence_parser.add_argument("--sequence-test-size", type=int, default=5000)
    m4_sequence_parser.add_argument("--seed", type=int, default=13)
    m4_sequence_parser.add_argument("--paths", type=Path, default=None)

    m4_rqk_parser = m4_subparsers.add_parser("rq-kmeans", help="Train RQ-KMeans code retrieval baseline.")
    m4_rqk_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m4_rqk_parser.add_argument("--data-root", type=Path, default=None)
    m4_rqk_parser.add_argument("--output-root", type=Path, default=None)
    m4_rqk_parser.add_argument("--num-levels", type=int, default=4)
    m4_rqk_parser.add_argument("--codebook-size", type=int, default=256)
    m4_rqk_parser.add_argument("--iterations", type=int, default=25)
    m4_rqk_parser.add_argument("--feature-dim", type=int, default=2048)
    m4_rqk_parser.add_argument("--device", default=None)
    m4_rqk_parser.add_argument("--swanlab-project", default="SkillRQ-M4")
    m4_rqk_parser.add_argument("--swanlab-run-name", default=None)
    m4_rqk_parser.add_argument("--disable-swanlab", action="store_true")
    m4_rqk_parser.add_argument("--paths", type=Path, default=None)

    m4_rqvae_parser = m4_subparsers.add_parser("rq-vae", help="Train ordinary RQ-VAE code retrieval baseline.")
    m4_rqvae_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m4_rqvae_parser.add_argument("--data-root", type=Path, default=None)
    m4_rqvae_parser.add_argument("--output-root", type=Path, default=None)
    m4_rqvae_parser.add_argument("--epochs", type=int, default=10)
    m4_rqvae_parser.add_argument("--batch-size", type=int, default=1024)
    m4_rqvae_parser.add_argument("--learning-rate", type=float, default=1e-3)
    m4_rqvae_parser.add_argument("--feature-dim", type=int, default=2048)
    m4_rqvae_parser.add_argument("--latent-dim", type=int, default=256)
    m4_rqvae_parser.add_argument("--num-levels", type=int, default=4)
    m4_rqvae_parser.add_argument("--codebook-size", type=int, default=256)
    m4_rqvae_parser.add_argument("--commitment-weight", type=float, default=0.25)
    m4_rqvae_parser.add_argument("--device", default=None)
    m4_rqvae_parser.add_argument("--swanlab-project", default="SkillRQ-M4")
    m4_rqvae_parser.add_argument("--swanlab-run-name", default=None)
    m4_rqvae_parser.add_argument("--disable-swanlab", action="store_true")
    m4_rqvae_parser.add_argument("--paths", type=Path, default=None)

    m5_parser = subparsers.add_parser("m5", help="Train residual coverage selector.")
    m5_subparsers = m5_parser.add_subparsers(dest="m5_command")
    m5_prepare_parser = m5_subparsers.add_parser("prepare", help="Prepare residual coverage supervision data.")
    m5_prepare_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m5_prepare_parser.add_argument("--model-kind", choices=["coverage", "code-plan"], default="coverage")
    m5_prepare_parser.add_argument("--m4-data-root", type=Path, default=None)
    m5_prepare_parser.add_argument("--m4-prediction-path", type=Path, default=None)
    m5_prepare_parser.add_argument("--output-root", type=Path, default=None)
    m5_prepare_parser.add_argument("--max-steps", type=int, default=6)
    m5_prepare_parser.add_argument("--limit-queries", type=int, default=None)
    m5_prepare_parser.add_argument("--paths", type=Path, default=None)

    m5_train_parser = m5_subparsers.add_parser("train", help="Train residual selector with coverage supervision.")
    m5_train_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m5_train_parser.add_argument("--model-kind", choices=["coverage", "code-plan"], default="coverage")
    m5_train_parser.add_argument("--data-root", type=Path, default=None)
    m5_train_parser.add_argument("--output-root", type=Path, default=None)
    m5_train_parser.add_argument("--epochs", type=int, default=10)
    m5_train_parser.add_argument("--batch-size", type=int, default=512)
    m5_train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    m5_train_parser.add_argument("--embedding-dim", type=int, default=512)
    m5_train_parser.add_argument("--hidden-dim", type=int, default=1024)
    m5_train_parser.add_argument("--coverage-weight", type=float, default=1.0)
    m5_train_parser.add_argument("--role-weight", type=float, default=0.3)
    m5_train_parser.add_argument("--stop-weight", type=float, default=0.3)
    m5_train_parser.add_argument("--max-vocab-size", type=int, default=200000)
    m5_train_parser.add_argument("--device", default=None)
    m5_train_parser.add_argument("--swanlab-project", default="SkillRQ-M5")
    m5_train_parser.add_argument("--swanlab-run-name", default=None)
    m5_train_parser.add_argument("--disable-swanlab", action="store_true")
    m5_train_parser.add_argument("--paths", type=Path, default=None)

    m5_predict_parser = m5_subparsers.add_parser("predict", help="Predict residual code paths.")
    m5_predict_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m5_predict_parser.add_argument("--model-kind", choices=["coverage", "code-plan"], default="coverage")
    m5_predict_parser.add_argument("--m4-data-root", type=Path, default=None)
    m5_predict_parser.add_argument("--m4-prediction-path", type=Path, default=None)
    m5_predict_parser.add_argument("--checkpoint-root", type=Path, required=True)
    m5_predict_parser.add_argument("--output-root", type=Path, default=None)
    m5_predict_parser.add_argument("--max-steps", type=int, default=6)
    m5_predict_parser.add_argument("--top-n-paths", type=int, default=8)
    m5_predict_parser.add_argument("--candidates-per-step", type=int, default=20)
    m5_predict_parser.add_argument("--stop-threshold", type=float, default=0.55)
    m5_predict_parser.add_argument("--split", default=None)
    m5_predict_parser.add_argument("--device", default=None)
    m5_predict_parser.add_argument("--swanlab-project", default="SkillRQ-M5")
    m5_predict_parser.add_argument("--swanlab-run-name", default=None)
    m5_predict_parser.add_argument("--disable-swanlab", action="store_true")
    m5_predict_parser.add_argument("--enable-exact-first-retrieval", action="store_true")
    m5_predict_parser.add_argument("--disable-m4-candidate-prior", action="store_true")
    m5_predict_parser.add_argument("--paths", type=Path, default=None)

    m5_eval_parser = m5_subparsers.add_parser("evaluate", help="Evaluate residual coverage predictions.")
    m5_eval_parser.add_argument("--prediction-path", type=Path, required=True)
    m5_eval_parser.add_argument("--output-path", type=Path, required=True)
    m5_eval_parser.add_argument("--top-k", default="5,10,20,50,100")
    m5_eval_parser.add_argument("--set-metric-name", default="tool_set_recall")

    m7_parser = subparsers.add_parser("m7", help="Train and run role-aware sequence-aware reranker.")
    m7_subparsers = m7_parser.add_subparsers(dest="m7_command")

    m7_prepare_parser = m7_subparsers.add_parser("prepare", help="Prepare reranker supervision data.")
    m7_prepare_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m7_prepare_parser.add_argument("--m4-data-root", type=Path, default=None)
    m7_prepare_parser.add_argument("--output-root", type=Path, default=None)
    m7_prepare_parser.add_argument("--negatives-per-positive", type=int, default=2)
    m7_prepare_parser.add_argument("--limit-queries", type=int, default=None)
    m7_prepare_parser.add_argument("--paths", type=Path, default=None)

    m7_train_parser = m7_subparsers.add_parser("train", help="Train M7 reranker.")
    m7_train_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m7_train_parser.add_argument("--model-kind", choices=["standard", "code-aware"], default="standard")
    m7_train_parser.add_argument("--data-root", type=Path, default=None)
    m7_train_parser.add_argument("--output-root", type=Path, default=None)
    m7_train_parser.add_argument("--epochs", type=int, default=10)
    m7_train_parser.add_argument("--batch-size", type=int, default=512)
    m7_train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    m7_train_parser.add_argument("--embedding-dim", type=int, default=512)
    m7_train_parser.add_argument("--hidden-dim", type=int, default=1024)
    m7_train_parser.add_argument("--max-vocab-size", type=int, default=300000)
    m7_train_parser.add_argument("--role-weight", type=float, default=0.2)
    m7_train_parser.add_argument("--stage-weight", type=float, default=0.2)
    m7_train_parser.add_argument("--order-weight", type=float, default=0.2)
    m7_train_parser.add_argument("--code-consistency-weight", type=float, default=0.3)
    m7_train_parser.add_argument("--schema-weight", type=float, default=0.2)
    m7_train_parser.add_argument("--coverage-gain-weight", type=float, default=0.2)
    m7_train_parser.add_argument("--prompt-usefulness-weight", type=float, default=0.3)
    m7_train_parser.add_argument("--device", default=None)
    m7_train_parser.add_argument("--swanlab-project", default="SkillRQ-M7")
    m7_train_parser.add_argument("--swanlab-run-name", default=None)
    m7_train_parser.add_argument("--disable-swanlab", action="store_true")
    m7_train_parser.add_argument("--paths", type=Path, default=None)

    m7_joint_train_parser = m7_subparsers.add_parser(
        "joint-train",
        help="Jointly train residual code prediction and reranking ablation model.",
    )
    m7_joint_train_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m7_joint_train_parser.add_argument("--data-root", type=Path, default=None)
    m7_joint_train_parser.add_argument("--output-root", type=Path, default=None)
    m7_joint_train_parser.add_argument("--epochs", type=int, default=20)
    m7_joint_train_parser.add_argument("--batch-size", type=int, default=512)
    m7_joint_train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    m7_joint_train_parser.add_argument("--embedding-dim", type=int, default=512)
    m7_joint_train_parser.add_argument("--hidden-dim", type=int, default=1024)
    m7_joint_train_parser.add_argument("--code-embedding-dim", type=int, default=128)
    m7_joint_train_parser.add_argument("--max-vocab-size", type=int, default=300000)
    m7_joint_train_parser.add_argument("--code-weight", type=float, default=1.0)
    m7_joint_train_parser.add_argument("--role-weight", type=float, default=0.2)
    m7_joint_train_parser.add_argument("--stage-weight", type=float, default=0.2)
    m7_joint_train_parser.add_argument("--order-weight", type=float, default=0.2)
    m7_joint_train_parser.add_argument("--soft-code-weight", type=float, default=0.1)
    m7_joint_train_parser.add_argument("--enable-shared-encoder", action="store_true")
    m7_joint_train_parser.add_argument("--enable-soft-code-distribution", action="store_true")
    m7_joint_train_parser.add_argument("--device", default=None)
    m7_joint_train_parser.add_argument("--swanlab-project", default="SkillRQ-M7")
    m7_joint_train_parser.add_argument("--swanlab-run-name", default=None)
    m7_joint_train_parser.add_argument("--disable-swanlab", action="store_true")
    m7_joint_train_parser.add_argument("--paths", type=Path, default=None)

    m7_predict_parser = m7_subparsers.add_parser("predict", help="Rerank M4/M5 candidate predictions.")
    m7_predict_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m7_predict_parser.add_argument("--model-kind", choices=["standard", "code-aware"], default=None)
    m7_predict_parser.add_argument("--prediction-path", type=Path, required=True)
    m7_predict_parser.add_argument("--m4-data-root", type=Path, default=None)
    m7_predict_parser.add_argument("--checkpoint-root", type=Path, required=True)
    m7_predict_parser.add_argument("--output-root", type=Path, default=None)
    m7_predict_parser.add_argument("--top-k", type=int, default=100)
    m7_predict_parser.add_argument("--device", default=None)
    m7_predict_parser.add_argument("--swanlab-project", default="SkillRQ-M7")
    m7_predict_parser.add_argument("--swanlab-run-name", default=None)
    m7_predict_parser.add_argument("--disable-swanlab", action="store_true")
    m7_predict_parser.add_argument("--paths", type=Path, default=None)

    m7_joint_predict_parser = m7_subparsers.add_parser("joint-predict", help="Predict with a joint M7 ablation model.")
    m7_joint_predict_parser.add_argument("--target", choices=["capability", "skill"], required=True)
    m7_joint_predict_parser.add_argument("--prediction-path", type=Path, required=True)
    m7_joint_predict_parser.add_argument("--m4-data-root", type=Path, default=None)
    m7_joint_predict_parser.add_argument("--checkpoint-root", type=Path, required=True)
    m7_joint_predict_parser.add_argument("--output-root", type=Path, default=None)
    m7_joint_predict_parser.add_argument("--top-k", type=int, default=100)
    m7_joint_predict_parser.add_argument("--device", default=None)
    m7_joint_predict_parser.add_argument("--swanlab-project", default="SkillRQ-M7")
    m7_joint_predict_parser.add_argument("--swanlab-run-name", default=None)
    m7_joint_predict_parser.add_argument("--disable-swanlab", action="store_true")
    m7_joint_predict_parser.add_argument("--paths", type=Path, default=None)

    m7_eval_parser = m7_subparsers.add_parser("evaluate", help="Evaluate M7 reranked predictions.")
    m7_eval_parser.add_argument("--prediction-path", type=Path, required=True)
    m7_eval_parser.add_argument("--output-path", type=Path, required=True)
    m7_eval_parser.add_argument("--top-k", default="5,10,20,50,100")
    m7_eval_parser.add_argument("--set-metric-name", default="tool_set_recall")

    prompt_parser = subparsers.add_parser("prompt", help="Build LLM agent planning prompts.")
    prompt_subparsers = prompt_parser.add_subparsers(dest="prompt_command")
    prompt_build_parser = prompt_subparsers.add_parser("build", help="Build code-path-guided planning prompts.")
    prompt_build_parser.add_argument("--prediction-path", type=Path, required=True)
    prompt_build_parser.add_argument("--m5-prediction-path", type=Path, default=None)
    prompt_build_parser.add_argument("--output-root", type=Path, required=True)
    prompt_build_parser.add_argument("--top-tools-per-step", type=int, default=3)
    prompt_build_parser.add_argument("--max-steps", type=int, default=6)
    prompt_build_parser.add_argument("--hide-scores", action="store_true")

    agent_sim_parser = subparsers.add_parser("agent-sim", help="Simulate and evaluate LLM tool-use plans.")
    agent_sim_subparsers = agent_sim_parser.add_subparsers(dest="agent_sim_command")
    agent_sim_mock_parser = agent_sim_subparsers.add_parser("mock", help="Run prompt-grounded mock tool-call simulation.")
    agent_sim_mock_parser.add_argument("--prompt-record-path", type=Path, required=True)
    agent_sim_mock_parser.add_argument("--output-root", type=Path, required=True)
    agent_sim_mock_parser.add_argument("--max-calls", type=int, default=6)
    agent_sim_mock_parser.add_argument("--tools-per-step", type=int, default=1)

    agent_sim_vllm_parser = agent_sim_subparsers.add_parser("vllm", help="Run vLLM tool-call planning inference.")
    agent_sim_vllm_parser.add_argument("--prompt-record-path", type=Path, required=True)
    agent_sim_vllm_parser.add_argument("--output-root", type=Path, required=True)
    agent_sim_vllm_parser.add_argument("--model", required=True)
    agent_sim_vllm_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    agent_sim_vllm_parser.add_argument("--dtype", default="auto")
    agent_sim_vllm_parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    agent_sim_vllm_parser.add_argument("--max-model-len", type=int, default=None)
    agent_sim_vllm_parser.add_argument("--trust-remote-code", action="store_true")
    agent_sim_vllm_parser.add_argument("--temperature", type=float, default=0.0)
    agent_sim_vllm_parser.add_argument("--top-p", type=float, default=1.0)
    agent_sim_vllm_parser.add_argument("--max-tokens", type=int, default=512)
    agent_sim_vllm_parser.add_argument("--batch-size", type=int, default=32)
    agent_sim_vllm_parser.add_argument("--limit", type=int, default=None)
    agent_sim_vllm_parser.add_argument("--seed", type=int, default=0)

    agent_sim_eval_parser = agent_sim_subparsers.add_parser("evaluate", help="Evaluate simulated tool-call plans.")
    agent_sim_eval_parser.add_argument("--plan-path", type=Path, required=True)
    agent_sim_eval_parser.add_argument("--output-path", type=Path, required=True)
    agent_sim_eval_parser.add_argument("--top-k", default="1,3,5,10")

    diagnostics_parser = subparsers.add_parser("diagnostics", help="Run upper-bound and attribution diagnostics.")
    diagnostics_subparsers = diagnostics_parser.add_subparsers(dest="diagnostics_command")
    diagnostics_run_parser = diagnostics_subparsers.add_parser("run", help="Run all diagnostics.")
    diagnostics_run_parser.add_argument("--target", choices=["capability", "skill"], default="capability")
    diagnostics_run_parser.add_argument("--project-root", type=Path, default=Path("."))
    diagnostics_run_parser.add_argument("--output-root", type=Path, default=None)
    diagnostics_run_parser.add_argument("--top-k", default="5,10,20,50,100")
    diagnostics_run_parser.add_argument("--no-joint-predictions", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "config" and args.config_command == "show":
        config = load_paths_config(args.path)
        print(json.dumps(config.to_json_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "data" and args.data_command == "build":
        paths = load_paths_config(args.paths)
        if args.dataset != "skillret":
            parser.error("M1 currently supports only --dataset skillret")
        stats = build_skillret_processed_data(paths)
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "capability" and args.capability_command == "build":
        paths = load_paths_config(args.paths)
        stats = build_capability_processed_data(
            paths,
            dataset=args.dataset,
            output_root=args.output_root,
            include_answer_trees=not args.skip_answer_trees,
            limit_tools=args.limit_tools,
            limit_queries=args.limit_queries,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "m2" and args.m2_command == "run":
        paths = load_paths_config(args.paths)
        summary = run_m2_baselines(
            paths,
            datasets=args.datasets,
            methods=args.methods,
            top_ks=_parse_top_k(args.top_k),
            max_queries=_none_if_zero(args.max_queries),
            max_candidates=_none_if_zero(args.max_candidates),
            run_root=args.run_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "m3" and args.m3_command == "build":
        paths = load_paths_config(args.paths)
        summary = build_m3_codebooks(
            paths,
            datasets=args.datasets,
            limit_per_dataset=args.limit_per_dataset,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.command == "m4":
        if args.m4_command == "prepare":
            paths = load_paths_config(args.paths)
            stats = prepare_m4_data(
                paths,
                target=args.target,
                datasets=args.datasets,
                output_root=args.output_root,
                limit_queries=args.limit_queries,
            )
            print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m4_command == "train":
            paths = load_paths_config(args.paths)
            data_root = args.data_root or paths.processed_root / "m4" / args.target
            if args.model_kind == "soft-multipath":
                output_root = args.output_root or paths.run_root / "m4_query_to_code" / "soft_multipath" / args.target
                summary = train_soft_multipath_code_predictor(
                    data_root=data_root,
                    output_root=output_root,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    code_embedding_dim=args.code_embedding_dim,
                    max_vocab_size=args.max_vocab_size,
                    contrastive_weight=args.contrastive_weight,
                    hierarchy_weight=args.hierarchy_weight,
                    path_bce_weight=args.path_bce_weight,
                    contrastive_negative_count=args.contrastive_negative_count,
                    temperature=args.temperature,
                    device=args.device,
                    swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                    swanlab_run_name=args.swanlab_run_name,
                    enable_exact_first_retrieval=args.enable_exact_first_retrieval,
                    use_m4_candidate_prior=not args.disable_m4_candidate_prior,
                )
            else:
                output_root = args.output_root or paths.run_root / "m4_query_to_code" / "capabilityrq" / args.target
                summary = train_capabilityrq(
                    data_root=data_root,
                    output_root=output_root,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    max_vocab_size=args.max_vocab_size,
                    device=args.device,
                    swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                    swanlab_run_name=args.swanlab_run_name,
                )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m4_command == "predict":
            paths = load_paths_config(args.paths)
            data_root = args.data_root or paths.processed_root / "m4" / args.target
            if args.model_kind == "soft-multipath":
                output_root = args.output_root or paths.run_root / "m4_query_to_code" / "predictions" / "soft_multipath" / args.target
                summary = predict_soft_multipath_codes(
                    data_root=data_root,
                    checkpoint_root=args.checkpoint_root,
                    output_root=output_root,
                    top_n_paths=args.top_n_paths,
                    candidate_budget=args.candidate_budget,
                    split=args.split,
                    beam_width=args.beam_width,
                    score_blend=args.score_blend,
                    device=args.device,
                    swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                    swanlab_run_name=args.swanlab_run_name,
                )
            else:
                output_root = args.output_root or paths.run_root / "m4_query_to_code" / "predictions" / args.target
                summary = predict_query_codes(
                    data_root=data_root,
                    checkpoint_root=args.checkpoint_root,
                    output_root=output_root,
                    top_n_paths=args.top_n_paths,
                    candidate_budget=args.candidate_budget,
                    split=args.split,
                    device=args.device,
                    swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                    swanlab_run_name=args.swanlab_run_name,
                )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m4_command == "evaluate":
            metrics = evaluate_m4_predictions(
                prediction_path=args.prediction_path,
                output_path=args.output_path,
                top_ks=_parse_top_k(args.top_k),
                set_metric_name=args.set_metric_name,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m4_command == "sequence-split":
            paths = load_paths_config(args.paths)
            data_root = args.data_root or paths.processed_root / "m4" / args.target
            output_root = args.output_root or paths.processed_root / "m4_sequence_eval" / args.target
            stats = build_sequence_eval_view(
                m4_data_root=data_root,
                output_root=output_root,
                sequence_dev_size=args.sequence_dev_size,
                sequence_test_size=args.sequence_test_size,
                seed=args.seed,
            )
            print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m4_command == "rq-kmeans":
            paths = load_paths_config(args.paths)
            data_root = args.data_root or paths.processed_root / "m4" / args.target
            output_root = args.output_root or paths.run_root / "m4_query_to_code" / "rq_kmeans" / args.target
            summary = train_rq_kmeans(
                data_root=data_root,
                output_root=output_root,
                num_levels=args.num_levels,
                codebook_size=args.codebook_size,
                iterations=args.iterations,
                feature_dim=args.feature_dim,
                device=args.device,
                swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                swanlab_run_name=args.swanlab_run_name,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m4_command == "rq-vae":
            paths = load_paths_config(args.paths)
            data_root = args.data_root or paths.processed_root / "m4" / args.target
            output_root = args.output_root or paths.run_root / "m4_query_to_code" / "rq_vae" / args.target
            summary = train_rq_vae(
                data_root=data_root,
                output_root=output_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                feature_dim=args.feature_dim,
                latent_dim=args.latent_dim,
                num_levels=args.num_levels,
                codebook_size=args.codebook_size,
                commitment_weight=args.commitment_weight,
                device=args.device,
                swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                swanlab_run_name=args.swanlab_run_name,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

    if args.command == "m5":
        if args.m5_command == "prepare":
            paths = load_paths_config(args.paths)
            m4_data_root = args.m4_data_root or paths.processed_root / "m4" / args.target
            if args.model_kind == "code-plan":
                output_root = args.output_root or paths.processed_root / "m5_code_plan" / args.target
                stats = prepare_code_path_planning_data(
                    m4_data_root=m4_data_root,
                    output_root=output_root,
                    m4_prediction_path=args.m4_prediction_path,
                    max_steps=args.max_steps,
                    limit_queries=args.limit_queries,
                )
            else:
                output_root = args.output_root or paths.processed_root / "m5" / args.target
                stats = prepare_m5_data(
                    m4_data_root=m4_data_root,
                    output_root=output_root,
                    max_steps=args.max_steps,
                    limit_queries=args.limit_queries,
                )
            print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m5_command == "train":
            paths = load_paths_config(args.paths)
            if args.model_kind == "code-plan":
                data_root = args.data_root or paths.processed_root / "m5_code_plan" / args.target
                output_root = args.output_root or paths.run_root / "m5_code_path_planner" / args.target
                summary = train_code_path_planner(
                    data_root=data_root,
                    output_root=output_root,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    coverage_weight=args.coverage_weight,
                    role_weight=args.role_weight,
                    stop_weight=args.stop_weight,
                    max_vocab_size=args.max_vocab_size,
                    device=args.device,
                    swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                    swanlab_run_name=args.swanlab_run_name,
                )
            else:
                data_root = args.data_root or paths.processed_root / "m5" / args.target
                output_root = args.output_root or paths.run_root / "m5_residual_selector" / args.target
                summary = train_residual_selector(
                    data_root=data_root,
                    output_root=output_root,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    coverage_weight=args.coverage_weight,
                    max_vocab_size=args.max_vocab_size,
                    device=args.device,
                    swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                    swanlab_run_name=args.swanlab_run_name,
                )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m5_command == "predict":
            paths = load_paths_config(args.paths)
            m4_data_root = args.m4_data_root or paths.processed_root / "m4" / args.target
            if args.model_kind == "code-plan":
                output_root = args.output_root or paths.run_root / "m5_code_path_planner" / "predictions" / args.target
                summary = predict_code_path_plan(
                    m4_data_root=m4_data_root,
                    checkpoint_root=args.checkpoint_root,
                    output_root=output_root,
                    m4_prediction_path=args.m4_prediction_path,
                    max_steps=args.max_steps,
                    top_n_paths=args.top_n_paths,
                    candidates_per_step=args.candidates_per_step,
                    stop_threshold=args.stop_threshold,
                    split=args.split,
                    device=args.device,
                    swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                    swanlab_run_name=args.swanlab_run_name,
                )
            else:
                output_root = args.output_root or paths.run_root / "m5_residual_selector" / "predictions" / args.target
                summary = predict_residual_paths(
                    data_root=m4_data_root,
                    checkpoint_root=args.checkpoint_root,
                    output_root=output_root,
                    max_steps=args.max_steps,
                    top_n_paths=args.top_n_paths,
                    candidates_per_step=args.candidates_per_step,
                    split=args.split,
                    device=args.device,
                    swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                    swanlab_run_name=args.swanlab_run_name,
                )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m5_command == "evaluate":
            metrics = evaluate_m5_predictions(
                prediction_path=args.prediction_path,
                output_path=args.output_path,
                top_ks=_parse_top_k(args.top_k),
                set_metric_name=args.set_metric_name,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

    if args.command == "m7":
        if args.m7_command == "prepare":
            paths = load_paths_config(args.paths)
            m4_data_root = args.m4_data_root or paths.processed_root / "m4" / args.target
            output_root = args.output_root or paths.processed_root / "m7" / args.target
            stats = prepare_m7_data(
                m4_data_root=m4_data_root,
                output_root=output_root,
                negatives_per_positive=args.negatives_per_positive,
                limit_queries=args.limit_queries,
            )
            print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m7_command == "train":
            paths = load_paths_config(args.paths)
            data_root = args.data_root or paths.processed_root / "m7" / args.target
            output_root = args.output_root or paths.run_root / "m7_reranker" / args.target
            summary = train_reranker(
                data_root=data_root,
                output_root=output_root,
                model_kind=args.model_kind,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                max_vocab_size=args.max_vocab_size,
                role_weight=args.role_weight,
                stage_weight=args.stage_weight,
                order_weight=args.order_weight,
                code_consistency_weight=args.code_consistency_weight,
                schema_weight=args.schema_weight,
                coverage_gain_weight=args.coverage_gain_weight,
                prompt_usefulness_weight=args.prompt_usefulness_weight,
                device=args.device,
                swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                swanlab_run_name=args.swanlab_run_name,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m7_command == "joint-train":
            paths = load_paths_config(args.paths)
            data_root = args.data_root or paths.processed_root / "m7" / args.target
            output_root = args.output_root or paths.run_root / "m7_joint_reranker" / args.target
            summary = train_joint_reranker(
                data_root=data_root,
                output_root=output_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                embedding_dim=args.embedding_dim,
                hidden_dim=args.hidden_dim,
                code_embedding_dim=args.code_embedding_dim,
                max_vocab_size=args.max_vocab_size,
                code_weight=args.code_weight,
                role_weight=args.role_weight,
                stage_weight=args.stage_weight,
                order_weight=args.order_weight,
                soft_code_weight=args.soft_code_weight,
                enable_shared_encoder=args.enable_shared_encoder,
                enable_soft_code_distribution=args.enable_soft_code_distribution,
                device=args.device,
                swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                swanlab_run_name=args.swanlab_run_name,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m7_command == "predict":
            paths = load_paths_config(args.paths)
            m4_data_root = args.m4_data_root or paths.processed_root / "m4" / args.target
            output_root = args.output_root or paths.run_root / "m7_reranker" / "predictions" / args.target
            summary = predict_reranked_capabilities(
                prediction_path=args.prediction_path,
                m4_data_root=m4_data_root,
                checkpoint_root=args.checkpoint_root,
                output_root=output_root,
                model_kind=args.model_kind,
                top_k=args.top_k,
                device=args.device,
                swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                swanlab_run_name=args.swanlab_run_name,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m7_command == "joint-predict":
            paths = load_paths_config(args.paths)
            m4_data_root = args.m4_data_root or paths.processed_root / "m4" / args.target
            output_root = args.output_root or paths.run_root / "m7_joint_reranker" / "predictions" / args.target
            summary = predict_joint_reranked_capabilities(
                prediction_path=args.prediction_path,
                m4_data_root=m4_data_root,
                checkpoint_root=args.checkpoint_root,
                output_root=output_root,
                top_k=args.top_k,
                device=args.device,
                swanlab_project=None if args.disable_swanlab else args.swanlab_project,
                swanlab_run_name=args.swanlab_run_name,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.m7_command == "evaluate":
            metrics = evaluate_reranked_predictions(
                prediction_path=args.prediction_path,
                output_path=args.output_path,
                top_ks=_parse_top_k(args.top_k),
                set_metric_name=args.set_metric_name,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

    if args.command == "prompt":
        if args.prompt_command == "build":
            summary = build_code_guided_prompts(
                prediction_path=args.prediction_path,
                m5_prediction_path=args.m5_prediction_path,
                output_root=args.output_root,
                top_tools_per_step=args.top_tools_per_step,
                max_steps=args.max_steps,
                include_scores=not args.hide_scores,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

    if args.command == "agent-sim":
        if args.agent_sim_command == "mock":
            summary = simulate_mock_tool_calls(
                prompt_record_path=args.prompt_record_path,
                output_root=args.output_root,
                max_calls=args.max_calls,
                tools_per_step=args.tools_per_step,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.agent_sim_command == "vllm":
            summary = run_vllm_tool_call_planning(
                prompt_record_path=args.prompt_record_path,
                output_root=args.output_root,
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                trust_remote_code=args.trust_remote_code,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
                limit=args.limit,
                seed=args.seed,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.agent_sim_command == "evaluate":
            metrics = evaluate_tool_call_plans(
                plan_path=args.plan_path,
                output_path=args.output_path,
                top_ks=_parse_top_k(args.top_k),
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

    if args.command == "diagnostics":
        if args.diagnostics_command == "run":
            summary = run_diagnostics(
                project_root=args.project_root.resolve(),
                target=args.target,
                output_root=args.output_root,
                top_ks=_parse_top_k(args.top_k),
                include_joint_predictions=not args.no_joint_predictions,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

    parser.print_help()
    return 0


def _parse_top_k(value: str) -> list[int]:
    top_ks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not top_ks:
        raise ValueError("--top-k must contain at least one positive integer")
    return top_ks


def _none_if_zero(value: int | None) -> int | None:
    if value == 0:
        return None
    return value
