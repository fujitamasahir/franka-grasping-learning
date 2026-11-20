"""Conditional observation-noise likelihood under a fixed CV state model.

Only noisy XY enters estimation. Q is specified, not jointly fitted with R.
Unknown accelerations/contact motion can be confounded with measurement noise.
"""
import numpy as np
from motion_aware_noise_estimator import validated_xy


def kalman_noise(noisy_xy, dt=.1, acceleration_std=.05):
    """Fit isotropic R=sigma² I by innovation likelihood, accounting for H P- H'.

    State per axis: [position, velocity]. Q = a_std² G G',
    G=[dt²/2,dt]. Independent Gaussian acceleration each control step.
    Condition on observation zero: position covariance R and velocity SD .1m/s.
    Return a point estimate AND model-conditional diagnostics, not validity proof.
    """
    y=validated_xy(noisy_xy)
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(acceleration_std) or acceleration_std<0:
        raise ValueError('Positive dt and nonnegative finite acceleration_std required')
    grid=np.linspace(0,.03,121);r=np.maximum(grid**2,1e-12)
    f=np.array([[1.,dt],[0.,1.]])
    g=np.array([dt*dt/2,dt]);q=acceleration_std**2*np.outer(g,g)
    count=len(grid);state=np.zeros((count,2,2));state[:,:,0]=y[0]
    covariance=np.zeros((count,2,2));covariance[:,0,0]=r;covariance[:,1,1]=.1**2
    nll=np.zeros(count);predvar=np.zeros(count);innovvar=np.zeros(count)
    for obs in y[1:]:
        state=state@f.T
        covariance=f@covariance@f.T+q
        projected=covariance[:,0,0].copy();s=projected+r
        residual=obs-state[:,:,0]
        nll+=np.log(s)+.5*np.sum(residual**2,axis=1)/s
        predvar+=projected;innovvar+=s
        gain=covariance[:,:,0]/s[:,None]
        state+=residual[:,:,None]*gain[:,None,:]
        # Joseph covariance update preserves PSD at low R.
        a=np.broadcast_to(np.eye(2),(count,2,2)).copy();a[:,:,0]-=gain
        covariance=a@covariance@a.transpose(0,2,1)+r[:,None,None]*gain[:,:,None]*gain[:,None,:]
    best=int(np.argmin(nll));support=grid[nll<=nll[best]+1.92]
    return {'sigma_m':float(grid[best]),'R_m2':float(grid[best]**2),
            'acceleration_std_m_s2':float(acceleration_std),'dt_s':float(dt),
            'mean_projected_state_prediction_variance_m2':float(predvar[best]/19),
            'mean_innovation_variance_m2':float(innovvar[best]/19),
            'likelihood_support_sigma_m':[float(support[0]),float(support[-1])],
            'grid_boundary':best in (0,count-1),'grid_step_m':.00025,
            'interpretation':'Conditional on fixed Q, Gaussian CV dynamics, isotropic independent sensor noise and initialization. Q/R cannot generally be identified from20 noisy positions. Contact/unknown acceleration invalidate sensor-noise attribution. Likelihood support is descriptive, not a calibrated confidence interval.'}
