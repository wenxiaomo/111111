Inputs:
    standardized samples X0; fixed ball count N_G; cluster count K;
    epsilon, tau, tau_r, delta, gamma_Sigma, lambda_1, lambda_H, lambda_2;
    outer-sweep budget T_max and tolerance eta_stop;
    configured reconstruction opportunities, budget R_G,
    reconstruction tolerance eta_GB, and patience s_GB.

Initialize:
    1. Apply Ball k-means to X0 to obtain N_G nonempty index sets I_i.
    2. Compute every descriptor s_i on X0.
    3. Initialize the autoencoder theta, prototype matrix H,
       strictly positive normalized triplets P with nonzero signed
       coefficients for at least some ball-cluster pairs, and
       Sigma >= delta I. Avoid the all-uniform triplet fixed point.
    4. Set reconstruction_attempts = 0 and initialize stopping counters.

For outer sweep t = 0, ..., T_max - 1:
    Hold the current partition and descriptors fixed for this sweep.

    A. Reweight at the current complete state:
       Encode all s_i and obtain latent centers and scale channels.
       Compute w_i, v_i, q_i, and omega_i.
       Set d_i = (1+epsilon)(q_i+2 epsilon^2)
                 / [2(q_i+epsilon^2)^(3/2)].
       Set b_i = (1+epsilon)q_i^2
                 / [2(q_i+epsilon^2)^(3/2)].
       Hold d_i and b_i fixed until the next outer sweep.
       Define the common surrogate Q_t by replacing
           sum_i omega_i * varphi_epsilon(q_i)
       with
           sum_i omega_i(theta) * (d_i*q_i + b_i)
       in the full objective. In particular, the intercept b_i
       remains inside the network-dependent term.

    B. Update the prototypes H:
       Using the latest state, set alpha_i = omega_i*d_i,
           W[i,j] = mu_ij - nu_ij,
           C[i,:] = latent center of ball i,
           A = diag(alpha_1, ..., alpha_N_G),
           M = W^T A W, and B = W^T A C.
       Solve M H + (lambda_H/2) H Sigma = B.
       Accept the feasible update only under the common Q_t
       nonincrease condition.

    C. Update all triplets P:
       For each ball i, choose beta_i > 0 satisfying
           beta_i * L_i <= 1,
           L_i = 4 alpha_i lambda_max(H Sigma^{-1} H^T).
       Repeat the prescribed number of assignment inner steps:
           Recompute w_i and v_i using the CURRENT inner triplets.
           For each cluster j, compute
               u_ij = 2 alpha_i H_j Sigma^{-1} v_i.
           For components r = (mu, nu, pi), set
               g_ij = (-u_ij, +u_ij, 0),
               z_ijr = [log(p_ijr) - beta_i*g_ijr]
                       / [1 + beta_i*lambda_1],
               p_ijr <- softmax_r(z_ijr).
       Retain the full three-component triplet for each (i,j).
       Check feasible Q_t nonincrease after the assignment block.

    D. Update the covariance:
       Recompute v_i with the new H and P; keep alpha_i fixed
       because theta has not changed in this block.
       Form S = sum_i alpha_i v_i v_i^T.
       Eigendecompose S = U diag(s_1, ..., s_d_b) U^T.
       Set Sigma <- U diag(max(delta, 2*s_k/gamma_Sigma)) U^T.
       Check feasible Q_t nonincrease.

    E. Update the autoencoder:
       Keep H, P, Sigma, d_i, and b_i fixed.
       Re-encode the current ball descriptors for each candidate theta.
       Differentiate the full network-dependent surrogate:
           J_rec(theta)
           + sum_i omega_i(theta) * [d_i*q_i(theta) + b_i]
           + (lambda_2/2) * ||P_phi theta||_2^2.
       Propose an SGD/gradient step. Accept a feasible step only
       when it passes the common Q_t descent test; otherwise
       backtrack or retain theta. Refresh d_i and b_i only in
       the next outer sweep.

    F. Evaluate the original objective J after the full sweep.

    G. If this is a scheduled reconstruction opportunity and
       reconstruction_attempts < R_G:
       Form the reconstruction candidate using the frozen
       single-point latent coordinates and overlap inheritance.
       Increment reconstruction_attempts.
       Evaluate J_candidate on the proposed partition and state.
       If J_candidate <= J_current:
           Accept the new partition, descriptors, and triplets;
           inherit theta, H, Sigma, and optimizer state.
           Update the low-gain stopping counter from the
           relative decrease in J.
       Else:
           Retain the entire current state and partition;
           increment the rejection counter.
       Stop reconstruction attempts when their budget is reached
       or the specified patience condition is met.

    H. Stop the outer loop if the prescribed sweep budget is
       reached or the relative change of the original objective
       is at most eta_stop.

Final labels:
    For each ball i, choose the smallest index maximizing
        w_ij = mu_ij - nu_ij over j = 1, ..., K.
    For each original sample k, output the label of its ball
    under the LAST ACCEPTED sample-to-ball mapping.

Outputs: final theta, H, P, Sigma, partition, and sample labels.
