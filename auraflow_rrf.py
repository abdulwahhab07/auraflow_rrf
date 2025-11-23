import torch
import math
from tqdm.auto import trange

@torch.no_grad()
def sample_auraflow_rrf(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """
    Relativistic Resonant Flow (RRF) Sampler.
    
    Implements a curvature-corrected Euler method with Tangential Damping and 
    High-Frequency Resonance Injection for Rectified Flow models.
    
    Args:
        model: The velocity prediction model v_theta.
        x: Initial latent state.
        sigmas: Schedule of noise levels (time steps).
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    
    # Constants definitions
    # Damping factor for the orthogonal component of the lookahead derivative.
    # Value < 1.0 mitigates manifold deviation (drift) caused by high CFG.
    TANGENTIAL_DAMPING = 0.85 
    
    # Threshold for resonance injection (tau). Detail enhancement activates
    # only after 20% of the sampling trajectory is complete.
    RESONANCE_START_PCT = 0.2

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma_curr
        
        # 1. First Order Euler Step (Probe)
        # Calculate velocity at current position: v_t
        denoised = model(x, sigma_curr * s_in, **extra_args)
        
        # Derive the time derivative. For Rectified Flow, v = (x_t - x_0) / sigma.
        # A stability check is applied to avoid division by zero near t=0.
        d_curr = (x - denoised) / sigma_curr if sigma_curr > 1e-4 else (x - denoised)

        # Predict future state estimate using explicit Euler
        x_pred = x + d_curr * dt
        
        # 2. Curvature Correction via Tangential Damping
        if sigma_next > 1e-4:
            # Evaluate velocity field at the predicted future position
            denoised_next = model(x_pred, sigma_next * s_in, **extra_args)
            d_next = (x_pred - denoised_next) / sigma_next
            
            # Geometric Decomposition:
            # Decompose d_next into components parallel and orthogonal to d_curr.
            # This handles the non-linear curvature induced by the vector field.
            
            # Flatten spatial dimensions for accurate dot product calculation
            d1_flat = d_curr.view(d_curr.shape[0], -1)
            d2_flat = d_next.view(d_next.shape[0], -1)
            
            # Compute projection scalar: (v_next . v_curr) / ||v_curr||^2
            dot_prod = torch.sum(d1_flat * d2_flat, dim=1, keepdim=True)
            norm_sq = torch.sum(d1_flat * d1_flat, dim=1, keepdim=True).clamp(min=1e-6)
            proj_scalar = dot_prod / norm_sq
            
            # Restore spatial dimensions for broadcasting
            proj_scalar = proj_scalar.view(-1, 1, 1, 1)
            
            # Calculate components
            d_next_parallel = proj_scalar * d_curr
            d_next_orthogonal = d_next - d_next_parallel
            
            # Apply Tangential Damping to the orthogonal component
            # mitigating oscillatory drift from the data manifold.
            d_next_corrected = d_next_parallel + (d_next_orthogonal * TANGENTIAL_DAMPING)
            
            # Compute final derivative using a weighted average (Approximated Crank-Nicolson)
            d_prime = (d_curr + d_next_corrected) * 0.5
        else:
            # Fallback to first-order update at the terminal step to ensure stability
            d_prime = d_curr
            
        # Update the latent state
        x_next = x + d_prime * dt
        
        # 3. Stochastic Resonance Injection (Spectral Preservation)
        # Re-injects high-frequency details lost due to discretization dissipation.
        
        if i < len(sigmas) - 2 and (1.0 - (sigma_curr / sigmas[0])) > RESONANCE_START_PCT:
            # Estimate current x_0 from the corrected trajectory
            x0_est = x_next - d_prime * sigma_next
            
            # Isolate high-frequency components via Difference of Gaussians (DoG)
            # approximated by subtracting a spatially averaged version of the signal.
            x0_blurred = torch.nn.functional.avg_pool2d(x0_est, kernel_size=3, stride=1, padding=1)
            high_freq = x0_est - x0_blurred
            
            # Calculate time-dependent resonance factor to simulate "focusing".
            # Scaling is proportional to the remaining noise level.
            resonance_factor = 0.05 * (1.0 - sigma_curr/sigmas[0])
            
            # Inject the high-frequency residual back into the latent state
            x_next = x_next + high_freq * resonance_factor

        x = x_next

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma_curr, 'sigma_hat': sigma_curr, 'denoised': denoised})

    return x
