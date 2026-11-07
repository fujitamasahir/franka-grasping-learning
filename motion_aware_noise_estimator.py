"""Noisy-observation-only estimators, with no simulator/metadata access."""
import numpy as np


def validated_xy(noisy_xy):
    a=np.asarray(noisy_xy,dtype=np.float64)
    if a.shape!=(20,2) or not np.isfinite(a).all():raise ValueError('20 finite noisy XY samples required')
    return a


def position_variance(noisy_xy):
    a=validated_xy(noisy_xy)
    return float(np.sqrt(np.var(a,axis=0,ddof=1).mean()))


def second_difference_mad(noisy_xy):
    a=validated_xy(noisy_xy);d=np.diff(a,n=2,axis=0)
    mad=np.median(np.abs(d-np.median(d,axis=0)),axis=0)
    sigma=mad/(0.6744897501960817*np.sqrt(6.))
    return float(np.sqrt(np.mean(sigma**2)))
