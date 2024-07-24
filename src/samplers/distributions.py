# from typing import Generator

# import numpy as np
# import torch
# from sklearn import datasets
# from torch.distributions.multivariate_normal import MultivariateNormal
# from torch.utils.data import DataLoader, TensorDataset

# from src.discrete_ot import OTPlanSampler


# class PairedSampler(Sampler):
#     def __init__(
#         self,
#         X_sampler: Sampler,
#         Y_sampler: Sampler,
#         batch_size: int = 128,
#         n_paired_samples: int | None = None,
#         m_unpaired_samples: int | None = None,
#         mini_batch_size: int | None = None,
#         otp_sampler: OTPlanSampler | None = None,
#         device: str = "cuda",
#     ):
#         super(PairedSampler, self).__init__(device=device)
#         self.batch_size = batch_size
#         self.device = device

#         if mini_batch_size is not None and otp_sampler is None:
#             raise ValueError("OTPlanSampler must initialized during mini-batch sampling! But is None.")
#         self.paired_loader = self._init_paired_loader(
#             X_sampler, Y_sampler, n_paired_samples, mini_batch_size, otp_sampler
#         )
#         self.paired_generator = iter(self.paired_loader)

#         self.unpaired_loader = self._init_unpaired_loader(X_sampler, Y_sampler, m_unpaired_samples)
#         self.unpaired_generator = iter(self.unpaired_loader)

#     def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
#         return self.sample_from_generator(self.unpaired_generator, self.unpaired_loader)

#     def sample_pair(self) -> tuple[torch.Tensor, torch.Tensor]:
#         return self.sample_from_generator(self.paired_generator, self.paired_loader)

#     def sample_from_generator(self, generator: Generator, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
#         with torch.no_grad():
#             try:
#                 X, Y = next(generator)
#             except StopIteration:
#                 generator = iter(loader)
#                 X, Y = next(generator)
#         return X, Y

#     def _init_paired_loader(
#         self,
#         X_sampler: Sampler,
#         Y_sampler: Sampler,
#         n_paired_samples: int,
#         mini_batch_size: int | None,
#         otp_sampler: OTPlanSampler | None,
#     ) -> DataLoader:
#         X_paired, Y_paired = torch.empty((0, X_sampler.dim)), torch.empty((0, Y_sampler.dim))
#         if mini_batch_size is not None:
#             num_sampling_iterations = n_paired_samples // mini_batch_size
#             num_remaining_samples = n_paired_samples % mini_batch_size
#             for _ in range(num_sampling_iterations):
#                 _X, _Y = X_sampler.sample(mini_batch_size), Y_sampler.sample(mini_batch_size)
#                 X, Y = otp_sampler.sample_plan(_X, _Y)
#                 X_paired, Y_paired = torch.cat((X_paired, X), 0), torch.cat((Y_paired, Y), 0)
#             if num_remaining_samples > 0:
#                 _X, _Y = X_sampler.sample(num_remaining_samples), Y_sampler.sample(num_remaining_samples)
#                 X, Y = otp_sampler.sample_plan(_X, _Y)
#                 X_paired, Y_paired = torch.cat((X_paired, X), 0), torch.cat((Y_paired, Y), 0)
#         else:
#             num_gaussians = len(X_sampler.mu)
#             X, Y = X_sampler.sample(n_paired_samples), Y_sampler.sample(n_paired_samples)
#             for i in range(num_gaussians):
#                 indices = np.arange(i, len(X), num_gaussians)
#                 for x in X[indices]:
#                     for y in Y[indices]:
#                         X_paired, Y_paired = torch.cat((X_paired, x[None, :]), 0), torch.cat((Y_paired, y[None, :]), 0)

#         dataset = TensorDataset(X_paired, Y_paired)
#         return DataLoader(
#             dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.device)
#         )

#     def _init_unpaired_loader(
#         self,
#         X_sampler: Sampler,
#         Y_sampler: Sampler,
#         m_unpaired_samples: int,
#     ) -> DataLoader:
#         X, Y = X_sampler.sample(m_unpaired_samples), Y_sampler.sample(m_unpaired_samples)
#         dataset = TensorDataset(X, Y)
#         return DataLoader(
#             dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.device)
#         )
