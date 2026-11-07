"""Dynamic rule correctness, boundary semantics and integrity rejection tests."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from action_chunking import BCActionChunkController
from evaluate_h4_adaptive_replanning import estimate_sigma
import evaluate_h4_dynamic_replanning as study
from h4_dynamic_controller import RollingReplanningRule, advance_at_boundary


class StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=0

    def forward(self,x):
        self.calls+=1
        xyz=torch.zeros((len(x),4,3));xyz[:,:,0]=torch.arange(4)+self.calls*10
        return xyz,torch.tensor([[1.,-1.,1.,-1.]]).repeat(len(x),1)


class FakeEnv:
    terminal_step=84

    def __init__(self,position_range):
        self.control_dt=.1;self.max_time=16.;self.max_move=.15;self.cube_id=0
        self.data=type('Data',(),{'xpos':np.array([[.55,.05,.425]])})()
        self.step_number=0;self.closed=False

    def obs(self):
        a=np.zeros(12,dtype=np.float32);a[3:6]=self.data.xpos[0];a[6:9]=a[3:6]
        return a

    def reset(self,seed):return self.obs(),{}

    def step(self,action):
        self.step_number+=1
        return self.obs(),0.,self.step_number>=self.terminal_step,False,{'lift_height':0.,'is_success':False}

    def close(self):self.closed=True


def checkpoint():
    return {'max_move':.15,'max_time':16.,'control_dt':.1,'obs_mean':torch.zeros(12),'obs_std':torch.ones(12)}


def scaled_window(sigma):
    x=np.arange(20,dtype=float);x=(x-x.mean())/x.std(ddof=1)
    return np.column_stack([x*sigma,-x*sigma])


class DynamicControllerTests(unittest.TestCase):
    def test_rule_has_no_noise_profile_sigma_or_transition_time_api(self):
        self.assertEqual(tuple(inspect.signature(RollingReplanningRule.observe).parameters),('self','noisy_xy'))
        self.assertEqual(tuple(inspect.signature(RollingReplanningRule.decide).parameters),('self',))
        self.assertEqual(tuple(inspect.signature(RollingReplanningRule).parameters),('online',))

    def test_exact_windows_no_current_or_future_sample_and_cutoff(self):
        rule=RollingReplanningRule(True);events=[]
        observations=np.column_stack([np.arange(90)*.0001,np.sin(np.arange(90))*.01])
        for t in range(90):
            e=rule.decide()
            if e:
                events.append(e)
                self.assertAlmostEqual(e['estimated_sigma_m'],estimate_sigma(observations[t-20:t]),places=15)
                self.assertEqual((e['window_start'],e['window_end']),(t-20,t-1))
            rule.observe(observations[t])
        self.assertEqual([e['step'] for e in events],list(range(20,81,4)))

    def test_initial_inclusive_threshold_and_strict_hysteresis(self):
        r=RollingReplanningRule(True)
        # Isolate exact threshold comparisons from floating sample variance.
        with patch('h4_dynamic_controller.estimate_sigma',return_value=.0075):
            for _ in range(20):r.observe([0,0])
            self.assertEqual(r.decide()['after'],4)
        for value,expected in ((.0085,4),(.0085001,1),(.0065,1),(.0064999,4)):
            for _ in range(4):r.observe([0,0])
            with patch('h4_dynamic_controller.estimate_sigma',return_value=value):
                self.assertEqual(r.decide()['after'],expected)

    def test_one_shot_only_one_decision_and_reset(self):
        r=RollingReplanningRule(False);events=[]
        for t in range(100):
            e=r.decide()
            if e:events.append(e)
            r.observe([float(t),0])
        self.assertEqual([e['step'] for e in events],[20])
        r.reset();self.assertEqual((len(r.xy),r.interval,r.events),(0,4,[]))

    def test_bidirectional_switch_consumes_old_chunk_and_refreshes_index_zero(self):
        m=StubModel();c=BCActionChunkController(m,np.zeros(12),np.ones(12),4,4)
        o=np.zeros(12);c.reset(o)
        for _ in range(3):c.advance(o)
        self.assertEqual(c.chunk_index,3)
        advance_at_boundary(c,o,{'after':1});self.assertEqual(c.chunk_index,0)
        self.assertEqual(c.planned_action[0],20)
        self.assertEqual(c.planned_action[3],1)
        for _ in range(3):c.advance(o)
        advance_at_boundary(c,o,{'after':4})
        self.assertEqual(c.chunk_index,0);self.assertEqual(c.planned_action[0],60)
        for expected in (1,2,3):
            c.advance(o);self.assertEqual(c.chunk_index,expected)
            self.assertEqual(c.planned_action[0],60+expected)
        c.reset(o);self.assertEqual(c.chunk_index,0);self.assertEqual(c.inference_calls,1)

    def test_partial_chunk_switch_rejected(self):
        c=BCActionChunkController(StubModel(),np.zeros(12),np.ones(12),4,4)
        c.reset(np.zeros(12))
        with self.assertRaises(RuntimeError):advance_at_boundary(c,np.zeros(12),{'after':1})

    def test_noise_changes_exactly_at_40_and_uses_common_absolute_rng(self):
        b,a,s=study.dynamic_noise(342000,'clean_to_noisy')
        b2,a2,s2=study.dynamic_noise(342000,'noisy_to_clean')
        b3,a3,s3=study.dynamic_noise(342000,'noisy')
        np.testing.assert_array_equal(b,b2);np.testing.assert_array_equal(b,b3)
        self.assertTrue(np.all(a[:40]==0));np.testing.assert_array_equal(a[40:],a3[40:])
        np.testing.assert_array_equal(a2[:40],a3[:40]);self.assertTrue(np.all(a2[40:]==0))
        self.assertEqual(s[39],0);self.assertEqual(s[40],np.float32(.01))
        np.testing.assert_array_equal(study.dynamic_noise(342000,'clean_to_noisy')[1],a)

    def test_ground_truth_window_diameter_and_mixed_rms_are_diagnostic_only(self):
        xy=np.zeros((60,2));gt=np.zeros((60,3));gt[59,0]=.002
        sigma=np.zeros(61);sigma[40:]=.01
        d=study.window_diagnostic(44,xy,gt,sigma)
        self.assertTrue(d['mixed_window']);self.assertAlmostEqual(d['injected_rms_sigma_m'],np.sqrt(.2)*.01)
        self.assertEqual(d['gt_xy_diameter_m'],0)
        d=study.window_diagnostic(60,xy,gt,sigma)
        self.assertFalse(d['mixed_window']);self.assertTrue(d['post_change_only'])
        self.assertEqual(d['gt_xy_diameter_m'],.002)
        self.assertEqual(d['estimated_sigma_m'],0)

    def fake_rollout(self,controller='Online'):
        with patch.object(study,'FrankaAutonomousEnv',FakeEnv):
            return study.rollout(StubModel(),checkpoint(),42,'clean',controller,342000,'a'*64,'b'*64,True)

    def test_terminal_boundary_has_no_extra_decision_or_inference(self):
        with patch.object(FakeEnv,'terminal_step',20):
            r=self.fake_rollout()
        self.assertEqual(r['decision_count'],0);self.assertEqual(r['bc_inference_calls'],5)
        self.assertEqual(len(json.loads(r['windows_json'])),1)
        self.assertEqual(r['steps'],20)

    def test_prefix_equality_switch_windows_and_resume_validation(self):
        rows=[self.fake_rollout(c) for c in ('R4','OneShot','Online')]
        self.assertEqual(len({r['prefix_actions_sha256'] for r in rows}),1)
        self.assertEqual(len({r['prefix_observations_sha256'] for r in rows}),1)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'r.csv'
            import csv
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=study.FIELDS);w.writeheader();w.writerow(rows[-1])
            r=study.read_rows(p)
            self.assertEqual(study.validate_rows(r,range(342000,342020),42,'clean','Online','a'*64,'b'*64),{342000})
            with self.assertRaises(ValueError):study.validate_rows(r+r,range(342000,342020),42,'clean','Online','a'*64,'b'*64)
            bad=dict(r[0]);bad['seed']='342001'
            with self.assertRaises(ValueError):study.validate_rows([bad],range(342000,342020),42,'clean','Online','a'*64,'b'*64)
            bad=dict(r[0]);bad['base_noise_sha256']='0'*64
            with self.assertRaises(ValueError):study.validate_rows([bad],range(342000,342020),42,'clean','Online','a'*64,'b'*64)
            self.assertEqual(len(json.loads(r[0]['decisions_json'])),16)

    def test_final_gate_refuses_failed_stationarity(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(study,'OUT',Path(tmp)):
            (Path(tmp)/'calibration_integrity_audit.json').write_text(json.dumps({'gate_passed':False,'protocol_sha256':'b'*64}))
            with self.assertRaises(RuntimeError):study.require_gate('final','b'*64)


if __name__=='__main__':unittest.main()
