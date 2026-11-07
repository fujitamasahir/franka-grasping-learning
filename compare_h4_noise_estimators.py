"""Bounded CLI for existing Stage1 JSONL observed_xy sequences; no simulation."""
import argparse
from itertools import islice
import json
from pathlib import Path
import numpy as np
from motion_aware_noise_estimator import position_variance,second_difference_mad
from kalman_noise_estimator import kalman_noise


def compare(input_path,limit=4,role='development'):
    if limit<1:raise ValueError('Explicit positive sequence limit required')
    rows=[];sequences=0
    with Path(input_path).open() as stream:
        for line in islice(stream,limit):
            record=json.loads(line);seed=int(record['seed'])
            if role=='final' and not 352000<=seed<=352019:
                raise ValueError('Reserved independent estimator Final seeds:352000..352019; development data prohibited')
            y=np.asarray(record['observed_xy'],dtype=float)
            if y.ndim!=2 or y.shape[1]!=2 or len(y)<20 or not np.isfinite(y).all():raise ValueError('Invalid observed_xy sequence')
            sequences+=1
            for t in range(20,len(y)+1,4):
                window=y[t-20:t]
                # Estimators receive only observations; metadata below is reporting only.
                k=kalman_noise(window)
                rows.append({'sequence_index':sequences-1,'step':t,'position_sigma_m':position_variance(window),
                             'MAD_sigma_m':second_difference_mad(window),'kalman':k})
    if not rows:raise ValueError('No diagnostic windows')
    return {'role':role,'sequences':sequences,'windows':len(rows),'sequence_limit':limit,
            'reserved_final_seeds':[352000,352019],'no_grasp_success_evaluation':True,
            'configuration':{'dt_s':.1,'acceleration_std_m_s2':.05,'sigma_grid_m':[0,.03,.00025],
                             'initial_velocity_std_m_s':.1,'parameters_fitted_on_existing_data':False},
            'mean_sigma_m':{'position':float(np.mean([r['position_sigma_m'] for r in rows])),
                            'MAD':float(np.mean([r['MAD_sigma_m'] for r in rows])),
                            'kalman':float(np.mean([r['kalman']['sigma_m'] for r in rows]))},
            'kalman_boundary_windows':sum(r['kalman']['grid_boundary'] for r in rows),'window_results':rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int,default=4)
    parser.add_argument('--role',choices=('development','final'),default='development');a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=compare(a.input,a.limit,a.role)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='window_results'},indent=2))


if __name__=='__main__':main()
