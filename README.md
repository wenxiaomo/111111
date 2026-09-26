import torch
import torch.nn as nn


class CovarianceAwareDistributionLoss(nn.Module):


    def __init__(self, latent_dim, delta=1e-3, eps=1.0,
                 tau_r=1.0, gamma_sigma=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.delta = delta
        self.eps = eps
        self.tau_r = tau_r
        self.gamma_sigma = gamma_sigma

        self.L_param = nn.Parameter(torch.eye(latent_dim) * 0.1)

    def get_sigma(self):

        L = torch.tril(self.L_param)
        Sigma = L @ L.T + self.delta * torch.eye(
            self.latent_dim, device=self.L_param.device
        )
        return Sigma

    def mahalanobis(self, v):

        Sigma = self.get_sigma()

        L = torch.linalg.cholesky(Sigma)
        y = torch.linalg.solve_triangular(L, v.T, upper=False)
        q = (y ** 2).sum(dim=0)
        return q

    def robust_penalty(self, q):

        return (1.0 + self.eps) * q / torch.sqrt(q + self.eps ** 2)

    def scale_weight(self, r_b):

        return torch.exp(-(r_b ** 2) / (self.tau_r ** 2))

    def forward(self, v, r_b):

        q = self.mahalanobis(v)
        rho = self.robust_penalty(q)
        omega = self.scale_weight(r_b)
        Sigma = self.get_sigma()
        log_det = torch.logdet(Sigma)
        loss = (omega * rho).sum() + 0.5 * self.gamma_sigma * log_det
        return loss, {'q': q, 'rho': rho, 'omega': omega, 'log_det': log_det}

    @torch.no_grad()
    def update_sigma(self, alpha, v):


        vw = v * alpha[:, None]  # [N_G, d_b]
        S = vw.T @ v  # [d_b, d_b]
        S = 0.5 * (S + S.T)

        eigvals, eigvecs = torch.linalg.eigh(S)
        eigvals = torch.clamp(eigvals, min=0.0)
        new_eig = torch.clamp(2.0 * eigvals / self.gamma_sigma, min=self.delta)
        Sigma_new = eigvecs @ torch.diag(new_eig) @ eigvecs.T
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)

        L_new = torch.linalg.cholesky(Sigma_new)
        self.L_param.data = torch.tril(L_new)
        return Sigma_new
