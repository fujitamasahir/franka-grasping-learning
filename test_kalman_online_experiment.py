"""Small synthetic tests only: no MuJoCo rollout or Final data analysis."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import kalman_online_experiment as exp
from h4_kalman_online_rule import KalmanReplanningRule
from h4_dynamic_controller import advance_at_boundary
from test_h4_dynamic_replanning import FakeEnv,StubModel,checkpoint
from action_chunking import BCActionChunkController


class KalmanOnlineTests(unittest.TestCase):
    def p(self):return json.loads(exp.PROTOCOL.read_text())

    def test_windows_initial_equivalence_cutoff_and_observation_only_api(self):
        self.assertEqual(tuple(inspect.signature(KalmanReplanningRule.observe).parameters),('self','noisy_xy'))
        self.assertEqual(tuple(inspect.signature(KalmanReplanningRule.decide).parameters),('self',))
        online=KalmanReplanningRule(True);oneshot=KalmanReplanningRule(False);windows=[]
        def fake(x):windows.append(np.asarray(x).copy());return {'sigma_m':.01}
        with patch('h4_kalman_online_rule.kalman_noise',side_effect=fake):
            for t in range(100):
                a=online.decide();b=oneshot.decide()
                if t==20:self.assertEqual(a,b)
                if a:np.testing.assert_array_equal(windows[-1][:,0],np.arange(t-20,t))
                online.observe([t,0]);oneshot.observe([t,0])
        self.assertEqual([e['step'] for e in online.events],list(range(20,81,4)))
        self.assertEqual(len(oneshot.events),1)

    def test_hysteresis_both_directions_and_fresh_chunk(self):
        r=KalmanReplanningRule(True)
        for _ in range(20):r.observe([0,0])
        with patch('h4_kalman_online_rule.kalman_noise',return_value={'sigma_m':.0075}):self.assertEqual(r.decide()['after'],4)
        for value,want in ((.0085,4),(.0085001,1),(.0065,1),(.0064999,4)):
            for _ in range(4):r.observe([0,0])
            with patch('h4_kalman_online_rule.kalman_noise',return_value={'sigma_m':value}):self.assertEqual(r.decide()['after'],want)
        c=BCActionChunkController(StubModel(),np.zeros(12),np.ones(12),4,1);c.reset(np.zeros(12))
        advance_at_boundary(c,np.zeros(12),{'after':4});self.assertEqual(c.chunk_index,0);self.assertEqual(c.inference_calls,2)

    def test_noise_absolute_step40_new_seed_and_paired_profiles(self):
        a,x,s=exp.dynamic_noise(990000,'clean_to_noisy');b,y,t=exp.dynamic_noise(990000,'noisy_to_clean')
        np.testing.assert_array_equal(a,b);self.assertTrue(np.all(x[:40]==0));self.assertTrue(np.all(y[40:]==0))
        self.assertGreater(np.abs(x[40:]).max(),0);np.testing.assert_array_equal(x[40:],(a*np.float32(.01))[40:])
        old,_=exp.make_noise_schedule(990000,1.,20261106,160);self.assertFalse(np.array_equal(a,old))
        self.assertEqual(s[39],0);self.assertEqual(s[40],np.float32(.01))

    def test_fake_rollout_terminal_no_extra_inference_and_legacy_restore(self):
        old=exp.legacy.RollingReplanningRule
        with exp.reuse(),patch.object(exp.legacy,'FrankaAutonomousEnv',FakeEnv),patch.object(FakeEnv,'terminal_step',20):
            row=exp.legacy.rollout(StubModel(),checkpoint(),42,'clean_to_noisy','Online',990000,'a'*64,'b'*64,True)
            self.assertEqual(row['decision_count'],0);self.assertEqual(row['bc_inference_calls'],5)
            strings={k:str(v) for k,v in row.items()}
            exp.legacy.validate_rows([strings],range(990000,990001),42,'clean_to_noisy','Online','a'*64,'b'*64)
        self.assertIs(exp.legacy.RollingReplanningRule,old)

    def test_resume_validation_rejects_duplicates_and_provenance(self):
        with exp.reuse(),patch.object(exp.legacy,'FrankaAutonomousEnv',FakeEnv):
            row=exp.legacy.rollout(StubModel(),checkpoint(),42,'noisy_to_clean','Online',990000,'a'*64,'b'*64)
            row={k:str(v) for k,v in row.items()}
            args=(range(990000,990002),42,'noisy_to_clean','Online','a'*64,'b'*64)
            exp.legacy.validate_rows([row],*args)
            with self.assertRaises(ValueError):exp.legacy.validate_rows([row,row],*args)
            bad=dict(row);bad['checkpoint_sha256']='c'*64
            with self.assertRaises(ValueError):exp.legacy.validate_rows([bad],*args)

    def test_final_and_analysis_require_complete_passing_gate(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(exp,'OUT',Path(tmp)):
            with patch.object(exp.legacy,'rollout') as rollout:
                with self.assertRaises(FileNotFoundError):exp.run('final',self.p(),'x')
                rollout.assert_not_called()
            with self.assertRaises(FileNotFoundError):exp.analyze(self.p(),'x')
            (Path(tmp)/'smoke_audit.json').write_text(json.dumps({'phase':'smoke','protocol_sha256':'x','gate_passed':False}))
            with self.assertRaises(RuntimeError):exp.verify_saved_audit('smoke',self.p(),'x')

    def test_fake_deadline_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(exp,'OUT',Path(tmp)),patch.object(exp,'cutoff',return_value=True),patch.object(exp.legacy,'check_model',return_value=(None,None,'a'*64)),patch.object(exp.legacy,'rollout') as rollout:
            exp.run('smoke',self.p(),exp.sha256(exp.PROTOCOL));rollout.assert_not_called()
            self.assertFalse(list(Path(tmp).glob('*.csv')))
            journal=json.loads((Path(tmp)/'runtime_journal.jsonl').read_text());self.assertEqual(journal['reason'],'hard_deadline')

    def test_summary_pairing_bootstrap_and_no_claim_when_CI_contains_zero(self):
        p=self.p();p['statistics']['bootstrap_replicates']=200
        data={}
        for m in exp.MODELS:
            for profile in exp.PROFILES:
                for c in exp.CONTROLLERS:
                    data[m,profile,c]=[{'success':v,'bc_inference_calls':10,'bc_total_inference_ms':1,'episode_wall_seconds':1,'steps':84,'steps_R1':0,'steps_R4':84,'decision_count':0,'decisions_json':'[]','switch_R4_to_R1':0,'switch_R1_to_R4':0} for v in (0,0,1,1)]
                for i,v in enumerate((1,1,0,1)):data[m,profile,'Online'][i]['success']=v
        s=exp.summarize(data,p)
        self.assertEqual(s['primary_online_minus_oneshot_pp'],25.)
        self.assertLessEqual(s['paired_bootstrap_95pct_CI_pp'][0],0)
        self.assertFalse(s['clear_improvement'])
        self.assertTrue(all(r['beneficial']-r['harmful']==1 for r in s['paired_comparisons']))
        self.assertEqual(len(s['paired_comparisons']),18);json.dumps(s,allow_nan=False)

    def test_mocked_complete_audit_has_no_stationarity_gate_and_detects_mutation(self):
        import csv
        p=self.p();p['evaluation']['smoke']=[990000,990000];ph=exp.sha256(exp.PROTOCOL)
        with tempfile.TemporaryDirectory() as tmp,patch.object(exp,'OUT',Path(tmp)),patch.object(exp.legacy,'FrankaAutonomousEnv',FakeEnv):
            for stem,cfg in exp.cells('smoke',p):
                with exp.reuse():
                    row=exp.legacy.rollout(StubModel(),checkpoint(),cfg['model_seed'],cfg['profile'],cfg['controller'],990000,cfg['checkpoint_sha256'],ph,example=True)
                exp.write_json(Path(tmp)/f'{stem}.json',cfg)
                with (Path(tmp)/f'{stem}.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=exp.legacy.FIELDS);writer.writeheader();writer.writerow(row)
            result=exp.audit('smoke',p,ph)
            self.assertTrue(result['gate_passed']);self.assertEqual(result['complete_cells'],24)
            self.assertTrue(result['stationarity_not_an_acceptance_gate']);self.assertEqual(result['prefix_pairing_checks'],6)
            stem,cfg=next(exp.cells('smoke',p));metadata=Path(tmp)/f'{stem}.json'
            changed=dict(cfg);changed['noise_generation_seed']=0;metadata.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):exp.audit('smoke',p,ph)

    def test_seeds_scale_and_Kalman_source_frozen(self):
        p,ph=exp.protocol()
        self.assertEqual(p['evaluation']['expected_episodes'],{'smoke':240,'final':24000})
        self.assertFalse(set(range(*[p['evaluation']['smoke'][0],p['evaluation']['smoke'][1]+1]))&set(range(361000,362000)))
        self.assertNotIn('stationarity_gate',p);self.assertEqual(len(list(exp.cells('smoke',p))),24)
        self.assertFalse(exp.OUT.exists())


if __name__=='__main__':unittest.main()
