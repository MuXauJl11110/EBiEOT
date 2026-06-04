#!/usr/bin/env python3
"""Compare EBiEOT MNIST classification against Downloads/mnist_energy_grid.py."""


import importlib.util
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_REPO = Path(__file__).resolve().parents[1]
_REF_PATH = Path("/Users/michael/Downloads/mnist_energy_grid.py")

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_reference():
    spec = importlib.util.spec_from_file_location("mnist_energy_grid", _REF_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _map_ref_state_to_repo(ref_model, repo_model) -> None:
    """Copy reference CNNEmbeddingEnergy weights into ClassificationBasedEBiEOT."""
    repo_model.class_potential.logits.data.copy_(ref_model.class_potential.data)
    em = repo_model.energy_model
    rem = ref_model
    em.encoder.load_state_dict(rem.encoder.state_dict())
    em.feature_projector.load_state_dict(rem.feat_proj.state_dict())
    em.y_embed.load_state_dict(rem.y_embed.state_dict())
    em.head.load_state_dict(rem.energy_head.state_dict())


def _map_repo_state_to_ref(repo_model, ref_model) -> None:
    ref_model.class_potential.data.copy_(repo_model.class_potential.logits.data)
    rem = ref_model
    em = repo_model.energy_model
    rem.encoder.load_state_dict(em.encoder.state_dict())
    rem.feat_proj.load_state_dict(em.feature_projector.state_dict())
    rem.y_embed.load_state_dict(em.y_embed.state_dict())
    rem.energy_head.load_state_dict(em.head.state_dict())


def _build_loaders(ref_mod, data_root: str, seed: int, paired_pc: int, unpaired_pc: int):
    from src.utils.datasets.mnist_classification import build_mnist_datasets

    train_ds, test_ds, num_classes, _, _ = build_mnist_datasets(data_root)
    by_class = ref_mod.indices_by_class_from_dataset(train_ds, num_classes)
    val_pc = max(1, int(np.ceil(paired_pc * 0.25)))
    rng = random.Random(seed + 10_000 + 1)
    paired_idx, marginal_idx, val_idx = ref_mod.build_paired_marginal_val_indices(
        by_class, paired_pc, unpaired_pc, val_pc, rng
    )
    if not marginal_idx:
        marginal_idx = paired_idx.copy()

    def loader(indices, bs, shuffle):
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            Subset(train_ds, indices),
            batch_size=bs,
            shuffle=shuffle,
            generator=g if shuffle else None,
        )

    return (
        loader(paired_idx, 64, True),
        loader(marginal_idx, 128, True),
        loader(val_idx, 512, False),
        DataLoader(test_ds, batch_size=512, shuffle=False),
        paired_idx,
        marginal_idx,
        val_idx,
    )


def _run_ref_epoch(ref_mod, model, paired_loader, marginal_loader, device, config):
    from itertools import cycle

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    ema = ref_mod.EMA(model, decay=config["ema_decay"])
    epoch_steps = max(1, max(len(paired_loader), len(marginal_loader)))
    paired_iter = cycle(paired_loader)
    marginal_iter = cycle(marginal_loader)
    global_step = 0
    running = {"loss": 0.0, "joint": 0.0, "margy": 0.0, "logz": 0.0}

    model.train()
    for _ in range(epoch_steps):
        x_p, y_p = next(paired_iter)
        x_m, _ = next(marginal_iter)
        x_p, y_p, x_m = x_p.to(device), y_p.to(device), x_m.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.loss_with_parts(x_p, y_p, x_m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        ema.update(model, global_step)
        global_step += 1
        running["loss"] += float(loss.item())
        running["joint"] += parts["joint_term"]
        running["margy"] += parts["marg_y_term"]
        running["logz"] += parts["logz_term"]

    for k in running:
        running[k] /= epoch_steps
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    val_m = ref_mod.evaluate(model, paired_loader, device)  # placeholder replaced below
    return running, backup, ema, optimizer, global_step


def main() -> int:
    from src.ebieot.classification_based import ClassificationBasedEBiEOT
    from src.utils.datasets import mnist_classification as mc

    if not _REF_PATH.exists():
        print(f"Reference not found: {_REF_PATH}")
        return 1

    ref = _load_reference()
    data_root = str(_REPO / "data" / "mnist")
    seed = 53  # seed + run_id*11 for run_id=1
    paired_pc, unpaired_pc = 20, 0
    device = torch.device("cpu")
    config = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        "ema_decay": 0.999,
        "epsilon": 0.5,
    }

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    paired_ld, marg_ld, val_ld, test_ld, p_idx, m_idx, v_idx = _build_loaders(
        ref, data_root, seed, paired_pc, unpaired_pc
    )

    train_ds, _, nc, ic, isz = mc.build_mnist_datasets(data_root)
    by_class = mc.indices_by_class_from_dataset(train_ds, nc)
    rng = random.Random(seed + 10_000 + 1)
    p2, m2, v2 = mc.build_paired_marginal_val_indices(
        by_class, paired_pc, unpaired_pc, mc.val_per_class_from_paired(paired_pc, 0.25), rng
    )
    if not m_idx:
        m_idx = p_idx.copy()
    if not m2:
        m2 = p2.copy()

    print("=== Index splits ===")
    print(f"paired match: {p_idx == p2}")
    print(f"marginal match: {m_idx == m2}")
    print(f"val match: {v_idx == v2}")

    ref_model = ref.CNNEmbeddingEnergy(nc, ic, isz, epsilon=0.5).to(device)
    repo_model = ClassificationBasedEBiEOT(
        num_classes=nc,
        in_channels=ic,
        image_size=isz,
        epsilon=0.5,
        arch="cnn",
        hidden_dim=128,
    ).to(device)

    torch.manual_seed(0)
    ref_model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    torch.manual_seed(0)
    for mod in repo_model.modules():
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()

    _map_ref_state_to_repo(ref_model, repo_model)

    x0, y0 = next(iter(paired_ld))
    x0, y0 = x0.to(device), y0.to(device)
    with torch.no_grad():
        e_ref = ref_model.compute_energies(x0)
        e_repo = repo_model.compute_energies(x0)
        loss_ref, parts_ref = ref_model.loss_with_parts(x0, y0, x0)
        parts_repo = repo_model.loss_with_parts(x0, y0, x0)

    print("\n=== Forward (synced init) ===")
    print(f"energy max |diff|: {(e_ref - e_repo).abs().max().item():.3e}")
    print(f"loss |diff|: {abs(loss_ref.item() - parts_repo['loss'].item()):.3e}")
    print(
        f"parts joint/margy/logz |diff|: "
        f"{abs(parts_ref['joint_term'] - parts_repo['jointTerm'].item()):.3e}, "
        f"{abs(parts_ref['marg_y_term'] - parts_repo['margYTerm'].item()):.3e}, "
        f"{abs(parts_ref['logz_term'] - parts_repo['logzTerm'].item()):.3e}"
    )

    # One training epoch with identical batch order
    from itertools import cycle

    def epoch_metrics_ref(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ema = ref.EMA(model, decay=0.999)
        steps = max(1, max(len(paired_ld), len(marg_ld)))
        pi, mi = cycle(paired_ld), cycle(marg_ld)
        run = [0.0, 0.0, 0.0, 0.0]
        gs = 0
        model.train()
        for _ in range(steps):
            xp, yp = next(pi)
            xm, _ = next(mi)
            xp, yp, xm = xp.to(device), yp.to(device), xm.to(device)
            opt.zero_grad(set_to_none=True)
            loss, pr = model.loss_with_parts(xp, yp, xm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema.update(model, gs)
            gs += 1
            run[0] += loss.item()
            run[1] += pr["joint_term"]
            run[2] += pr["marg_y_term"]
            run[3] += pr["logz_term"]
        run = [v / steps for v in run]
        bk = {k: v.clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val = ref.evaluate(model, val_ld, device)
        model.load_state_dict(bk)
        test = ref.evaluate(model, test_ld, device)
        return run, val, test, bk

    def epoch_metrics_repo(model):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ema = mc.EMA(model, decay=0.999)
        metrics, gs = mc.train_classification_epoch(
            model, paired_ld, marg_ld, opt, device, ema, 0, 5.0
        )
        bk = {k: v.clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        val = mc.evaluate_classification(model, val_ld, device)
        model.load_state_dict(bk)
        test = mc.evaluate_classification(model, test_ld, device)
        return (
            [
                metrics["train_loss"],
                metrics["train_joint"],
                metrics["train_marg_y"],
                metrics["train_logz"],
            ],
            val,
            test,
            bk,
        )

    torch.manual_seed(seed)
    ref_model2 = ref.CNNEmbeddingEnergy(nc, ic, isz, epsilon=0.5).to(device)
    torch.manual_seed(seed)
    repo_model2 = ClassificationBasedEBiEOT(
        num_classes=nc, in_channels=ic, image_size=isz, epsilon=0.5, arch="cnn", hidden_dim=128
    ).to(device)

    r_run, r_val, r_test, r_bk = epoch_metrics_ref(ref_model2)
    torch.manual_seed(seed)
    repo_model3 = ClassificationBasedEBiEOT(
        num_classes=nc, in_channels=ic, image_size=isz, epsilon=0.5, arch="cnn", hidden_dim=128
    ).to(device)
    p_run, p_val, p_test, p_bk = epoch_metrics_repo(repo_model3)

    print("\n=== After 1 epoch (independent inits, same seed, same loaders) ===")
    print(f"train_loss ref={r_run[0]:.6f} repo={p_run[0]:.6f} diff={abs(r_run[0]-p_run[0]):.3e}")
    print(f"val_acc    ref={r_val['acc']:.6f} repo={p_val['acc']:.6f}")
    print(f"test_acc   ref={r_test['acc']:.6f} repo={p_test['acc']:.6f}")

    # Synced init full mini-run (separate model copies, same pre-epoch weights)
    torch.manual_seed(99)
    ref_model3 = ref.CNNEmbeddingEnergy(nc, ic, isz, epsilon=0.5).to(device)
    repo_model4 = ClassificationBasedEBiEOT(
        num_classes=nc, in_channels=ic, image_size=isz, epsilon=0.5, arch="cnn", hidden_dim=128
    ).to(device)
    for m in ref_model3.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    for m in repo_model4.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    _map_ref_state_to_repo(ref_model3, repo_model4)

    ref_model_sync = deepcopy(ref_model3)
    repo_model_sync = deepcopy(repo_model4)
    r_run2, r_val2, r_test2, _ = epoch_metrics_ref(ref_model_sync)
    p_run2, p_val2, p_test2, _ = epoch_metrics_repo(repo_model_sync)

    print("\n=== After 1 epoch (identical initial weights) ===")
    print(f"train_loss diff={abs(r_run2[0]-p_run2[0]):.3e}")
    print(f"val_acc diff={abs(r_val2['acc']-p_val2['acc']):.3e}")
    print(f"test_acc diff={abs(r_test2['acc']-p_test2['acc']):.3e}")

    tol = 1e-5
    ok = (
        (e_ref - e_repo).abs().max().item() < tol
        and p_idx == p2
        and m_idx == m2
        and v_idx == v2
    )
    print(f"\n{'PASS' if ok else 'FAIL'}: CNN forward + index-split parity")
    print(
        "Note: per-epoch train_loss may differ due to DataLoader shuffle; "
        "run scripts/run_mnist_energy_grid.py vs reference for end-to-end metrics."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
