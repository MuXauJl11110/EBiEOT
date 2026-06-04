import torch
from torch.distributions.distribution import Distribution


class SampleBuffer:
    """
    Replay buffer of (x, y) pairs with optional per-row replay (probability ``p``).

    **Devices:** ``buffer_x`` and ``buffer_y`` always live on CPU. Methods accept
    ``batch_x`` / ``batch_y`` on any device; values are copied to the buffer with
    ``.cpu()``. ``__call__`` returns ``batch_x`` as given, ``initial_y`` from
    ``noise_gen`` aligned to ``batch_x`` (device and dtype via ``.to(batch_x)``),
    and ``row_ids`` on ``batch_x.device``. Integer indices for replay are sampled
    on CPU then moved to ``batch_x.device`` for the returned ``row_ids`` tensor.
    """

    def __init__(
        self,
        noise_gen: Distribution,
        p: float = 0.95,
        max_samples: int = 10000,
    ):
        self.noise_gen = noise_gen
        self.p = p
        self.max_samples = max_samples

        self.buffer_x = torch.empty(0, device="cpu")
        self.buffer_y = torch.empty(0, device="cpu")

    def fresh_noise_init(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Sample initial y from ``noise_gen`` with the same batch size as ``x_batch``."""
        return self.noise_gen.sample((x_batch.size(0),)).to(x_batch)

    def append_pairs(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> None:
        """Append detached CPU copies of ``batch_x`` and ``batch_y``, then trim oldest rows if over capacity."""
        batch_x = batch_x.detach().cpu()
        batch_y = batch_y.detach().cpu()
        if self.buffer_x.numel() == 0:
            self.buffer_x = batch_x
            self.buffer_y = batch_y
        else:
            self.buffer_x = torch.cat((self.buffer_x, batch_x), dim=0)
            self.buffer_y = torch.cat((self.buffer_y, batch_y), dim=0)

        if self.buffer_x.size(0) > self.max_samples:
            excess = self.buffer_x.size(0) - self.max_samples
            self.buffer_x = self.buffer_x[excess:]
            self.buffer_y = self.buffer_y[excess:]

    def push(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        row_ids: torch.Tensor | None = None,
    ) -> None:
        """
        Update the buffer.

        * ``row_ids is None``: append all rows.
        * ``row_ids`` is a 1D long tensor (same length as ``batch_x``): rows with
          ``row_ids == -1`` are appended; rows with ``row_ids >= 0`` overwrite those
          buffer indices.
        """
        if row_ids is None:
            self.append_pairs(batch_x, batch_y)
            return

        row_ids = row_ids.to(device=batch_x.device, dtype=torch.long)
        replay_row_mask = row_ids >= 0
        if replay_row_mask.any():
            buffer_indices = (
                row_ids[replay_row_mask].detach().cpu().to(dtype=torch.long)
            )
            self.buffer_x[buffer_indices] = batch_x[replay_row_mask].detach().cpu()
            self.buffer_y[buffer_indices] = batch_y[replay_row_mask].detach().cpu()
        fresh_row_mask = ~replay_row_mask
        if fresh_row_mask.any():
            self.append_pairs(batch_x[fresh_row_mask], batch_y[fresh_row_mask])

    def __len__(self) -> int:
        return self.buffer_x.size(0)

    def __call__(
        self, batch_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build a batch aligned with input rows: with probability ``p`` per row, replay a
        random buffer pair; otherwise keep ``batch_x[i]`` and draw fresh noise for y.

        Returns ``(batch_x, initial_y, row_ids)`` where ``row_ids[i] >= 0`` is the buffer
        index used for replay and ``row_ids[i] == -1`` means that row was fresh noise
        (append on ``push``).
        """
        batch_size = batch_x.size(0)
        if len(self) < 1:
            row_ids = torch.full(
                (batch_size,), -1, dtype=torch.long, device=batch_x.device
            )
            return batch_x, self.fresh_noise_init(batch_x), row_ids

        replay_row_mask = torch.rand(batch_size, device=batch_x.device) < self.p
        num_replay_rows = int(replay_row_mask.sum().item())

        initial_y_batch = self.fresh_noise_init(batch_x)
        row_ids = torch.full((batch_size,), -1, dtype=torch.long, device=batch_x.device)

        if num_replay_rows > 0:
            buffer_row_count = self.buffer_x.size(0)
            sampled_buffer_row_indices = torch.randint(
                0, buffer_row_count, (num_replay_rows,), dtype=torch.long, device="cpu"
            )
            replay_y_batch = self.buffer_y[sampled_buffer_row_indices].to(
                batch_x.device
            )
            replay_buffer_indices = sampled_buffer_row_indices.to(batch_x.device)
            replay_row_indices = torch.where(replay_row_mask)[0]
            initial_y_batch[replay_row_indices] = replay_y_batch
            row_ids[replay_row_indices] = replay_buffer_indices

        return batch_x, initial_y_batch, row_ids
