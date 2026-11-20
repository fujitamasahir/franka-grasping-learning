import numpy as np
import torch
import unittest,tempfile
from pathlib import Path

from action_chunking import ActionChunkBC, BCActionChunkController, build_windows, pair_rollout_rows, validate_rollout_seed_set
from franka_chunk_aware_residual_env import FrankaChunkAwareResidualEnv
from evaluate_h4_replanning import FIELDS, exclusive_run_lock, validate_rows
from analyze_h4_replanning import exact_mcnemar, holm, paired_bootstrap_ci
from evaluate_h4_observation_noise import apply_observation_noise, make_noise_schedule, phase_seed_range
from franka_autonomous_env import FrankaAutonomousEnv


def test_windows_share_valid_starts_and_never_cross_episode():
    obs=np.zeros((14,12),np.float32);obs[:,9]=np.tile(np.arange(7)*.1,2)/16
    actions=np.zeros((14,4),np.float32);ids=np.repeat([1,2],7)
    x1,y1,_=build_windows(obs,actions,ids,[1,2],1)
    x4,y4,_=build_windows(obs,actions,ids,[1,2],4)
    assert len(x1)==len(x4)==8
    np.testing.assert_array_equal(x1,x4)
    assert y1.shape==(8,1,4) and y4.shape==(8,4,4)


def test_controller_executes_four_cached_actions_then_replans():
    class Stub(torch.nn.Module):
        def __init__(self): super().__init__();self.calls=0
        def forward(self,x):
            self.calls+=1
            return torch.full((len(x),4,3),self.calls/10),torch.full((len(x),4),-1.)
    m=Stub();c=BCActionChunkController(m,np.zeros(12,np.float32),np.ones(12,np.float32),4)
    first=c.reset(np.zeros(12,np.float32));assert first[0]==.1
    for i in range(3):
        c.advance(np.zeros(12,np.float32));assert c.planned_action[0]==.1
    assert c.chunk_index==3
    c.advance(np.zeros(12,np.float32));assert c.planned_action[0]==.2
    assert m.calls==2 and c.inference_calls==2


def test_controller_h1_replans_every_step_and_resume_keeps_plan():
    model=ActionChunkBC(1)
    c=BCActionChunkController(model,np.zeros(12,np.float32),np.ones(12,np.float32),1)
    c.reset(np.zeros(12,np.float32));planned=c.planned_action.copy();state=c.state_dict()
    d=BCActionChunkController(model,np.zeros(12,np.float32),np.ones(12,np.float32),1);d.load_state_dict(state)
    np.testing.assert_array_equal(d.planned_action,planned)
    d.advance(np.ones(12,np.float32));assert d.inference_calls==2


class ActionChunkingTests(unittest.TestCase):
    def test_paired_outcomes_require_identical_seed_positions(self):
        baseline=[{'seed':10,'success':0},{'seed':11,'success':1}]
        policy=[{'seed':10,'success':1},{'seed':11,'success':0}]
        self.assertEqual(len(pair_rollout_rows(baseline,policy,[10,11])),2)
        self.assertEqual(set(validate_rollout_seed_set(baseline,[10,11])),{10,11})
        with self.assertRaises(ValueError):pair_rollout_rows(baseline,policy[:1])
        with self.assertRaises(ValueError):pair_rollout_rows(baseline,policy,[10,12])
        with self.assertRaises(ValueError):validate_rollout_seed_set(baseline,[10,12])

    def test_data_windows_and_controller_timing(self):
        test_windows_share_valid_starts_and_never_cross_episode()
        test_controller_executes_four_cached_actions_then_replans()
        test_controller_h1_replans_every_step_and_resume_keeps_plan()

    def test_h4_runtime_replanning_intervals_and_gripper_alignment(self):
        class IndexedChunks(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.calls=0
            def forward(self, obs):
                self.calls += 1
                xyz=torch.tensor([[[float(self.calls*10+i),0.,0.] for i in range(4)]]).repeat(len(obs),1,1)
                grip=torch.tensor([[1.,-1.,1.,-1.]]).repeat(len(obs),1)
                return xyz,grip
        for interval, expected_indices in ((1,[0,0,0,0,0]),(2,[0,1,0,1,0]),(4,[0,1,2,3,0])):
            m=IndexedChunks(); c=BCActionChunkController(m,np.zeros(12),np.ones(12),4,replan_interval=interval)
            action=c.reset(np.zeros(12))
            seen=[]
            for step in range(5):
                seen.append((c.chunk_index,float(action[0]),float(action[3])))
                # A terminal transition must not cause another inference.
                if step == 4:
                    self.assertFalse(c.advance(np.zeros(12),terminated=True))
                    break
                replanned=c.advance(np.zeros(12))
                action=c.planned_action
                if replanned: self.assertEqual(c.chunk_index,0)
            self.assertEqual([x[0] for x in seen],expected_indices)
            for index, value, grip in seen:
                self.assertEqual(int(value)%10,index)
                self.assertEqual(grip,1.0 if index%2==0 else -1.0)
            self.assertEqual(m.calls,c.inference_calls)
            self.assertEqual(m.calls,(5+interval-1)//interval)

    def test_h4_default_interval_is_backward_compatible_and_state_checked(self):
        m=ActionChunkBC(4)
        legacy=BCActionChunkController(m,np.zeros(12),np.ones(12),4)
        self.assertEqual(legacy.replan_interval,4)
        legacy.reset(np.zeros(12));state=legacy.state_dict()
        restored=BCActionChunkController(m,np.zeros(12),np.ones(12),4)
        restored.load_state_dict(state)
        self.assertEqual(restored.replan_interval,4)
        wrong=BCActionChunkController(m,np.zeros(12),np.ones(12),4,replan_interval=2)
        with self.assertRaises(ValueError):wrong.load_state_dict(state)

    def test_h4_exact_paired_analysis_and_holm(self):
        baseline={i:{'success':int(i in (0,1,2))} for i in range(5)}
        comparison={0:{'success':1},1:{'success':0},2:{'success':0},3:{'success':1},4:{'success':1}}
        result=exact_mcnemar(baseline,comparison)
        self.assertEqual((result['baseline_success_comparison_failure'],result['baseline_failure_comparison_success']),(2,2))
        self.assertEqual(result['exact_two_sided_p'],1.0)
        self.assertEqual(holm([.01,.04,.03]),[.03,.06,.06])
        ci=paired_bootstrap_ci(baseline,comparison,123)
        self.assertEqual(ci,paired_bootstrap_ci(baseline,comparison,123))

    def test_h4_resume_rows_validate_seed_order_and_provenance(self):
        def row(seed):
            d={k:'0' for k in FIELDS}
            d.update({'seed':str(seed),'model_seed':'42','replan_interval':'2','success':'1','steps':'10',
                'return':'1.0','cube_x':'0.55','cube_y':'0.0','region':'EN','closed_fraction':'0.5',
                'bc_inference_calls':'5','bc_total_inference_ms':'1.0','bc_mean_latency_ms':'0.2',
                'episode_wall_seconds':'0.1','checkpoint_sha256':'checkpoint','protocol_sha256':'protocol'})
            return d
        self.assertEqual(validate_rows([row(10)],range(10,12),42,2,'checkpoint','protocol'),{10})
        with self.assertRaises(ValueError):validate_rows([row(10),row(10)],range(10,12),42,2,'checkpoint','protocol')
        with self.assertRaises(ValueError):validate_rows([row(11),row(10)],range(10,12),42,2,'checkpoint','protocol')
        with self.assertRaises(ValueError):validate_rows([row(10)],range(10,12),42,2,'different','protocol')

    def test_h4_evaluator_rejects_concurrent_writer_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock=Path(tmp)/'eval.lock'
            with exclusive_run_lock(lock):
                self.assertTrue(lock.exists())
                with self.assertRaises(RuntimeError):
                    with exclusive_run_lock(lock): pass
            self.assertFalse(lock.exists())

    def test_observation_noise_changes_only_cube_and_relative_xy(self):
        obs=np.arange(12,dtype=np.float32)
        noisy=apply_observation_noise(obs,np.array([.01,-.02],np.float32))
        expected=np.array([3,4,6,7])
        np.testing.assert_array_equal(np.flatnonzero(noisy!=obs),expected)
        self.assertAlmostEqual(float(noisy[3]-obs[3]),.01,places=6)
        self.assertAlmostEqual(float(noisy[6]-obs[6]),.01,places=6)
        self.assertAlmostEqual(float(noisy[4]-obs[4]),-.02,places=6)
        self.assertAlmostEqual(float(noisy[7]-obs[7]),-.02,places=6)
        np.testing.assert_array_equal(apply_observation_noise(obs,np.zeros(2,np.float32)),obs)
        np.testing.assert_array_equal(obs,np.arange(12,dtype=np.float32))

    def test_noise_schedule_is_seed_step_indexed_and_paired_across_conditions(self):
        base_a,n5_a=make_noise_schedule(321042,.005,20261106,160)
        base_b,n10_b=make_noise_schedule(321042,.010,20261106,160)
        base_other,_=make_noise_schedule(321043,.005,20261106,160)
        np.testing.assert_array_equal(base_a,base_b)
        np.testing.assert_allclose(n10_b,2*n5_a,rtol=1e-6,atol=1e-9)
        self.assertFalse(np.array_equal(base_a,base_other))
        self.assertEqual(base_a.shape,(161,2))
        self.assertFalse(np.array_equal(base_a[4],base_a[5]))

    def test_noise_evaluator_reads_preregistered_phase_ranges(self):
        ranges={'smoke_seed_range_inclusive':[322000,322019],'final_seed_range_inclusive':[321000,321999]}
        self.assertEqual(phase_seed_range(ranges,'smoke'),[322000,322019])
        self.assertEqual(phase_seed_range(ranges,'final'),[321000,321999])
        with self.assertRaises(KeyError):phase_seed_range(ranges,'validation')

    def test_policy_observation_noise_does_not_change_environment_state(self):
        env=FrankaAutonomousEnv(position_range=.20)
        try:
            obs,_=env.reset(seed=321100)
            qpos=env.data.qpos.copy();cube=env.data.xpos[env.cube_id].copy()
            noisy=apply_observation_noise(obs,np.array([.01,-.01],np.float32))
            np.testing.assert_array_equal(env.data.qpos,qpos)
            np.testing.assert_array_equal(env.data.xpos[env.cube_id],cube)
            self.assertEqual(float(noisy[5]),float(obs[5]))
            self.assertEqual(float(noisy[10]),float(obs[10]))
            self.assertEqual(float(noisy[11]),float(obs[11]))
        finally: env.close()

    def test_wrapper_alignment_xyz_only_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'h4.pt';model=ActionChunkBC(4)
            torch.save({'model_state_dict':model.state_dict(),'obs_mean':torch.zeros(12),'obs_std':torch.ones(12),
                'horizon':4,'max_move':.15,'max_time':16.0},path)
            env=FrankaChunkAwareResidualEnv(path,.20,.005,'uniform',.12,.90,'shaped',.005)
            try:
                obs,_=env.reset(seed=290042)
                np.testing.assert_array_equal(obs[12:16],env.controller.planned_action)
                self.assertEqual(obs[16],0.)
                state=env.training_state_dict();next1,r1,t1,tr1,i1=env.step(np.zeros(3,np.float32))
                self.assertAlmostEqual(float(next1[16]),1/3)
                np.testing.assert_array_equal(i1['executed_action'][3],i1['bc_action'][3])
                np.testing.assert_allclose(i1['applied_residual_xyz_m'],0,atol=1e-8)
                env.load_training_state_dict(state);next2,r2,t2,tr2,i2=env.step(np.zeros(3,np.float32))
                np.testing.assert_array_equal(next1,next2);self.assertEqual((r1,t1,tr1),(r2,t2,tr2))
                np.testing.assert_array_equal(i1['executed_action'],i2['executed_action'])
            finally:env.close()

