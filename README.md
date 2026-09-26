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


        import torch
import torch.nn as nn
import torch.nn.functional as F


class GBAutoEncoder(nn.Module):

    def __init__(self, feat_dim, hidden_dims=(128, 64), latent_dim=32):
        super().__init__()

        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.center_dim = latent_dim - 2
        assert self.center_dim >= 1

        input_dim = feat_dim + 2

        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
            prev = h
        enc_layers += [nn.Linear(prev, latent_dim), nn.LayerNorm(latent_dim), nn.GELU()]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        dec_layers += [nn.Linear(prev, feat_dim)]
        self.decoder_center = nn.Sequential(*dec_layers)

        h_r = max(16, latent_dim // 4)
        self.decoder_radius = nn.Sequential(
            nn.Linear(latent_dim, h_r), nn.GELU(),
            nn.Linear(h_r, 1), nn.Softplus()
        )
        self.decoder_eta = nn.Sequential(
            nn.Linear(latent_dim, h_r), nn.GELU(),
            nn.Linear(h_r, 1), nn.Sigmoid()
        )

    def encode(self, s):
        """s: [N_G, d0+2] -> z: [N_G, latent_dim]。"""
        return self.encoder(s)

    def split_latent(self, z):
        c_b = z[:, :self.center_dim]
        r_b = F.softplus(z[:, self.center_dim])
        eta_b = torch.sigmoid(z[:, self.center_dim + 1])
        return c_b, r_b, eta_b

    def decode(self, z):
        c_hat = self.decoder_center(z)
        r_hat = self.decoder_radius(z).squeeze(-1)
        eta_hat = self.decoder_eta(z).squeeze(-1)
        return c_hat, r_hat, eta_hat

    def forward(self, s):
        z = self.encode(s)
        c_b, r_b, eta_b = self.split_latent(z)
        c_hat, r_hat, eta_hat = self.decode(z)
        return {
            'z': z, 'c_b': c_b, 'r_b': r_b, 'eta_b': eta_b,
            'c_hat': c_hat, 'r_hat': r_hat, 'eta_hat': eta_hat,
        }

    @staticmethod
    def reconstruction_loss(out, s):
        c = s[:, :-2]
        r = s[:, -2]
        eta = s[:, -1]
        loss_c = 0.5 * F.mse_loss(out['c_hat'], c, reduction='sum')
        loss_r = 0.5 * F.mse_loss(out['r_hat'], r, reduction='sum')
        loss_e = 0.5 * F.mse_loss(out['eta_hat'], eta, reduction='sum')
        return loss_c + loss_r + loss_e
