"""
Estructura ResNet-18 CIFAR: recorte de canales muertos para reducir parámetros en RAM.

    La poda IMP es no estructurada; aquí se eliminan **solo** canales de salida cuyos
    pesos en ``fc`` son nulos en toda la columna (característica nunca usada), y se
    propaga el recorte hacia atrás por ``layer4`` (último bloque identidad y bloque
    con ``downsample``). Canales con un peso no nulo en ``fc`` no se tocan.

    El recorte de tensores en ``layer4`` altera el mapa frente al modelo original salvo
    que los canales eliminados sean realmente nulos en todo el tramo residual; para
    pruebas, anula esos canales en los pesos antes de llamar a esta función.

Si no hay columnas muertas en ``fc``, se devuelve el mismo grafo y ``state_dict``.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from pia.models.resnet_cifar import build_resnet18_cifar


def parameter_and_buffer_bytes(module: nn.Module) -> int:
    """Bytes totales de parámetros y buffers registrados."""
    p = sum(x.numel() * x.element_size() for x in module.parameters())
    b = sum(x.numel() * x.element_size() for x in module.buffers())
    return p + b


def _fc_dead_input_columns(fc_weight: Tensor, eps: float) -> list[int]:
    """Índices j donde toda la columna ``fc_weight[:, j]`` es ~cero."""
    dead: list[int] = []
    for j in range(fc_weight.shape[1]):
        if float(fc_weight[:, j].abs().max().item()) <= eps:
            dead.append(j)
    return dead


def _keep_index(keep: list[int], device: torch.device) -> Tensor:
    """Índices ``long`` en el mismo dispositivo que el tensor a ``index_select``."""
    return torch.tensor(keep, dtype=torch.long, device=device)


def _slice_bn2d_tensors(sd: dict[str, Tensor], bn_prefix: str, keep: list[int]) -> None:
    """Recorta peso/sesgo y estadísticas de ``BatchNorm2d`` a ``len(keep)`` canales."""
    ref = sd[f"{bn_prefix}weight"]
    idx = _keep_index(keep, ref.device)
    for suffix in ("weight", "bias", "running_mean", "running_var"):
        k = f"{bn_prefix}{suffix}"
        if k in sd and sd[k] is not None and sd[k].numel() > 0:
            t = sd[k]
            idx_k = idx.to(t.device, non_blocking=True)
            sd[k] = t.index_select(0, idx_k).clone()


def _slice_identity_basic_block_sd(
    sd: dict[str, Tensor], prefix: str, keep: list[int]
) -> None:
    """
    Recorta un ``BasicBlock`` con ``inplanes == planes`` y atajo identidad.

    ``prefix`` termina en ``.`` (p.ej. ``layer4.1.``).
    """
    w1 = sd[f"{prefix}conv1.weight"]
    idx = _keep_index(keep, w1.device)
    sd[f"{prefix}conv1.weight"] = w1.index_select(0, idx).index_select(1, idx).clone()
    _slice_bn2d_tensors(sd, f"{prefix}bn1.", keep)
    w2 = sd[f"{prefix}conv2.weight"]
    sd[f"{prefix}conv2.weight"] = w2.index_select(0, idx).index_select(1, idx).clone()
    _slice_bn2d_tensors(sd, f"{prefix}bn2.", keep)


def _slice_downsample_basic_block_sd(
    sd: dict[str, Tensor], prefix: str, keep: list[int], *, inplanes: int
) -> None:
    """
    Recorta el primer bloque de ``layer4`` (stride 2 + ``downsample`` 1x1).

    ``conv1`` pasa de ``[planes, inplanes, 3, 3]`` a ``[K, inplanes, 3, 3]``;
    ``conv2`` a ``[K, K, 3, 3]``; ``downsample.0`` a ``[K, inplanes, 1, 1]``.
    """
    w1 = sd[f"{prefix}conv1.weight"]
    idx = _keep_index(keep, w1.device)
    sd[f"{prefix}conv1.weight"] = w1.index_select(0, idx).clone()
    _slice_bn2d_tensors(sd, f"{prefix}bn1.", keep)
    w2 = sd[f"{prefix}conv2.weight"]
    sd[f"{prefix}conv2.weight"] = w2.index_select(0, idx).index_select(1, idx).clone()
    _slice_bn2d_tensors(sd, f"{prefix}bn2.", keep)
    ds_w = sd[f"{prefix}downsample.0.weight"]
    sd[f"{prefix}downsample.0.weight"] = ds_w.index_select(0, idx).clone()
    _slice_bn2d_tensors(sd, f"{prefix}downsample.1.", keep)


def _basic_block_from_sd(
    sd: dict[str, Tensor],
    prefix: str,
    *,
    inplanes: int,
    planes: int,
    stride: int,
    downsample: nn.Module | None,
) -> nn.Module:
    """Construye un ``BasicBlock`` leyendo tensores bajo ``prefix``."""
    norm_layer = nn.BatchNorm2d

    class _BB(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                inplanes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = norm_layer(planes)
            self.downsample = downsample

        def forward(self, x: Tensor) -> Tensor:
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            out = self.relu(out)
            return out

    block = _BB()
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = block.load_state_dict(sub, strict=True)
    if missing or unexpected:
        msg = (
            f"load_state_dict BasicBlock {prefix}: "
            f"missing={missing} unexpected={unexpected}"
        )
        raise RuntimeError(msg)
    return block


def _downsample_from_sd(
    sd: dict[str, Tensor], prefix: str, *, inplanes: int, outplanes: int
) -> nn.Module:
    """``Sequential(Conv2d 1x1, BN)`` como en torchvision."""
    seq = nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm2d(outplanes),
    )
    sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    seq.load_state_dict(sub, strict=True)
    return seq


def slim_resnet18_cifar_from_state_dict(
    sd_in: dict[str, Tensor],
    *,
    eps: float = 1e-12,
    num_classes: int = 10,
) -> tuple[nn.Module, dict[str, Tensor]]:
    """
    Elimina columnas totalmente nulas de ``fc`` y recorta ``layer4`` en consecuencia.

    Solo se implementa la propagación por los dos bloques de ``layer4`` (ResNet-18
    CIFAR de torchvision). El resto de capas queda intacto.

    Args:
        sd_in: ``state_dict`` (CPU o GPU) compatible con ``build_resnet18_cifar``.
        eps: Umbral absoluto para considerar un peso cero.
        num_classes: Clases de la cabeza ``fc`` (10 en CIFAR-10).

    Returns:
        ``(modelo, state_dict)`` con menos parámetros si había columnas muertas;
        en caso contrario, un clon del modelo original y el mismo dict copiado.

    Raises:
        RuntimeError: Si el recorte deja ``fc`` sin columnas.
    """
    sd: dict[str, Tensor] = copy.deepcopy(sd_in)
    fc_w = sd["fc.weight"]
    if fc_w.shape[0] != num_classes:
        msg = (
            f"fc.weight.shape[0] debe ser num_classes={num_classes}, "
            f"obtuve {fc_w.shape[0]}."
        )
        raise ValueError(msg)
    c_in = fc_w.shape[1]
    dead = _fc_dead_input_columns(fc_w, eps)
    if not dead:
        model = build_resnet18_cifar(num_classes=num_classes)
        model.load_state_dict(sd)
        return model, sd

    keep = [j for j in range(c_in) if j not in dead]
    if not keep:
        msg = "Todas las columnas de fc están muertas; no se puede construir un modelo."
        raise RuntimeError(msg)

    sd["fc.weight"] = fc_w[:, keep].clone()
    if "fc.bias" in sd and sd["fc.bias"] is not None:
        pass

    _slice_identity_basic_block_sd(sd, "layer4.1.", keep)
    _slice_downsample_basic_block_sd(sd, "layer4.0.", keep, inplanes=c_in // 2)

    in_b1 = len(keep)
    ds = _downsample_from_sd(sd, "layer4.0.downsample.", inplanes=256, outplanes=in_b1)
    block0 = _basic_block_from_sd(
        sd, "layer4.0.", inplanes=256, planes=in_b1, stride=2, downsample=ds
    )
    block1 = _basic_block_from_sd(
        sd, "layer4.1.", inplanes=in_b1, planes=in_b1, stride=1, downsample=None
    )

    model = build_resnet18_cifar(num_classes=num_classes)
    model.layer4 = nn.Sequential(block0, block1)
    model.fc = nn.Linear(in_b1, num_classes, bias=True)
    model.fc.weight.data.copy_(sd["fc.weight"])
    if "fc.bias" in sd:
        model.fc.bias.data.copy_(sd["fc.bias"])

    tail_keys = {
        k
        for k in sd
        if k.startswith("layer4.") or k in ("fc.weight", "fc.bias")
    }
    rest = {k: v for k, v in sd.items() if k not in tail_keys}
    missing, unexpected = model.load_state_dict(rest, strict=False)
    if unexpected:
        msg = f"Cargando capas inferiores: claves inesperadas {unexpected}"
        raise RuntimeError(msg)
    if missing:
        allowed_missing = {
            k
            for k in missing
            if k.startswith("layer4.") or k.startswith("fc.")
        }
        bad = set(missing) - allowed_missing
        if bad:
            msg = f"Cargando capas inferiores: faltan {bad}"
            raise RuntimeError(msg)

    out_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model, out_sd


__all__ = ["parameter_and_buffer_bytes", "slim_resnet18_cifar_from_state_dict"]
