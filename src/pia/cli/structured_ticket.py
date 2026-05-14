"""
Entrada CLI para poda iterativa estructurada por ancho (CNN en cadena, CIFAR-10).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pia.data.cifar10 import build_cifar10_loaders
from pia.models.narrowable_chain_cnn import NarrowableChainCnn
from pia.models.resnet_cifar import apply_he_init
from pia.observability.logging_config import setup_pia_logging
from pia.pruning.structured_ticket import iterative_structured_magnitude_pruning

_log = logging.getLogger("pia.cli")


def main() -> None:
    """Parsea argumentos y ejecuta ``iterative_structured_magnitude_pruning``."""
    parser = argparse.ArgumentParser(
        description=(
            "Poda estructurada por recorte de ancho: NarrowableChainCnn sobre "
            "CIFAR-10 (sin reinicio a θ₀; rewind fijo a none)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs/structured_lt",
        help="Directorio base donde se crea --run-name.",
    )
    parser.add_argument("--run-name", type=str, default="struct_default")
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--prune-per-round", type=float, default=0.2)
    parser.add_argument("--epochs-per-round", type=int, default=10)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-weight", type=float, default=1e-5)
    parser.add_argument("--gamma-activation", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--initial-width",
        type=int,
        default=32,
        help="Canales de conv1/conv2 al inicio (debe ser >= 2 para podar).",
    )
    parser.add_argument(
        "--json-log",
        type=str,
        default="",
        help="Ruta opcional para log JSON del paquete pia.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="none",
        choices=("none", "cosine", "step"),
        help="Programación de LR por ronda (tras cada época).",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=10,
        help="StepLR: periodo en épocas (mínimo 1).",
    )
    parser.add_argument(
        "--lr-step-gamma",
        type=float,
        default=0.1,
        help="StepLR: factor multiplicativo.",
    )
    parser.add_argument(
        "--weight-l1-aggregation",
        type=str,
        default="sum",
        choices=("sum", "mean", "mean_per_param"),
        help="Agregación L1 de pesos en SparseLoss: sum, mean o mean_per_param.",
    )
    parser.add_argument(
        "--abort-on-val-acc-drop",
        type=float,
        default=-1.0,
        help=(
            "Si >= 0, detiene la fase actual si val/acc cae más que este delta "
            "entre épocas consecutivas."
        ),
    )
    parser.add_argument(
        "--early-stop-val-loss-relative",
        type=float,
        default=-1.0,
        help=(
            "Si >= 0, detiene la fase cuando val/loss supera (1+r) veces el mejor "
            "val/loss visto en esa fase durante --early-stop-patience épocas "
            "consecutivas (rebote tras un mínimo)."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=1,
        help=(
            "Con --early-stop-val-loss-relative >= 0: épocas consecutivas "
            "por encima del umbral."
        ),
    )
    parser.add_argument(
        "--early-stop-val-loss-min-delta",
        type=float,
        default=0.0,
        help="Mejora mínima en val/loss para registrar un nuevo mínimo en early stop.",
    )
    args = parser.parse_args()
    log_path = args.json_log.strip() or None
    setup_pia_logging(json_file=log_path)
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.initial_width < 2:
        parser.error("--initial-width debe ser >= 2.")
    if args.early_stop_val_loss_relative >= 0.0 and args.early_stop_patience < 1:
        parser.error(
            "--early-stop-patience debe ser >= 1 cuando early stop está activo."
        )
    modelo = NarrowableChainCnn(
        in_channels=3,
        width=args.initial_width,
        spatial_hw=(32, 32),
        num_classes=10,
    )
    apply_he_init(modelo)
    train_loader, val_loader = build_cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _log.info("Iniciando poda estructurada por ancho en %s", run_dir)
    abort_drop: float | None = (
        None if args.abort_on_val_acc_drop < 0 else args.abort_on_val_acc_drop
    )
    early_rel: float | None = (
        None
        if args.early_stop_val_loss_relative < 0.0
        else args.early_stop_val_loss_relative
    )
    iterative_structured_magnitude_pruning(
        run_dir=run_dir,
        num_rounds=args.num_rounds,
        prune_per_round=args.prune_per_round,
        epochs_per_round=args.epochs_per_round,
        lambda_weight=args.lambda_weight,
        gamma_activation=args.gamma_activation,
        data_root=args.data_root,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        lr_scheduler_kind=args.lr_scheduler,
        lr_step_size=args.lr_step_size,
        lr_step_gamma=args.lr_step_gamma,
        weight_l1_aggregation=args.weight_l1_aggregation,
        abort_training_on_val_acc_drop=abort_drop,
        early_stopping_val_loss_relative=early_rel,
        early_stopping_patience=args.early_stop_patience,
        early_stopping_min_delta=args.early_stop_val_loss_min_delta,
        custom_model=modelo,
        custom_train_loader=train_loader,
        custom_val_loader=val_loader,
    )


if __name__ == "__main__":
    main()
