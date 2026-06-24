import torch

# ----------------------------------------------------------------------
# Helper: build LR scheduler for one level
# ----------------------------------------------------------------------
def build_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str,
                    base_lr: float,
                    n_iters: int):
    """
    Create a PyTorch LR scheduler according to lr_schedule string.

    The base_lr is already set in optimizer; scheduler only controls
    multiplicative factor per step.
    """
    if scheduler_type == "none":
        return None

    if scheduler_type == "exp":
        # lr_t = base_lr * (lr_decay ** step)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.999
        )

    if scheduler_type == "cosine":
        # Cosine annealing from base_lr -> ~0 over n_iters
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, n_iters)
        )

    if scheduler_type == "linear":
        # Linearly decay from 1.0 -> 0.0 over n_iters-1 steps
        def lr_lambda(step: int):
            if n_iters <= 1:
                return 1.0
            t = min(step, n_iters - 1)
            return 1.0 - float(t) / float(n_iters - 1)

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda
        )

    raise ValueError(f"Unsupported lr_schedule: {lr_schedule}")