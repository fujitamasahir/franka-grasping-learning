"""Frozen H1/H4 BC controller with a stepwise, always-on SAC XYZ residual."""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from franka_autonomous_env import FrankaAutonomousEnv
from action_chunking import load_bc_policy, BCActionChunkController


class FrankaChunkAwareResidualEnv(gym.Wrapper):
    def __init__(self, bc_model, position_range=.20, residual_meters=.005,
                 sampling_mode='uniform', hard_inner_range=.12,
                 hard_fraction=.90, reward_mode='shaped',
                 residual_penalty_coefficient=.005):
        if sampling_mode not in ('uniform', 'hard') or reward_mode not in ('sparse', 'shaped'):
            raise ValueError('Invalid sampling/reward mode')
        if not 0 < residual_meters <= .03:
            raise ValueError('Invalid residual bound')
        self.bc_model, checkpoint = load_bc_policy(bc_model)
        self.bc_model.eval()
        for param in self.bc_model.parameters():
            param.requires_grad_(False)
        self.horizon = int(checkpoint['horizon'])
        self.obs_mean = checkpoint['obs_mean'].cpu().numpy().astype(np.float32)
        self.obs_std = checkpoint['obs_std'].cpu().numpy().astype(np.float32)
        self.controller = BCActionChunkController(self.bc_model, self.obs_mean, self.obs_std, self.horizon)
        base = FrankaAutonomousEnv(position_range=position_range)
        super().__init__(base)
        self.env.max_move=float(checkpoint['max_move']); self.env.max_time=float(checkpoint['max_time'])
        self.residual_meters=float(residual_meters); self.sampling_mode=sampling_mode
        self.hard_inner_range=float(hard_inner_range); self.hard_fraction=float(hard_fraction)
        self.reward_mode=reward_mode; self.residual_penalty_coefficient=float(residual_penalty_coefficient)
        self.action_space=spaces.Box(-1.,1.,shape=(3,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(17,),dtype=np.float32)
        self.current_obs=None; self.previous_distance=0.; self.previous_lift=0.; self.hard_sampled=False
        self.total_bc_latency=0.; self.bc_latencies=[]; self.bc_inference_calls=0

    def _sample_position(self):
        rng=self.env.np_random; outer=self.env.position_range; inner=self.hard_inner_range
        self.hard_sampled=bool(rng.random()<self.hard_fraction)
        if self.hard_sampled:
            while True:
                dx,dy=rng.uniform(-outer,outer,size=2)
                if abs(dx)>inner or abs(dy)>inner: break
        else: dx,dy=rng.uniform(-inner,inner,size=2)
        return float(.55+dx),float(dy)

    def _augmented_obs(self):
        phase=0. if self.horizon==1 else self.controller.chunk_index/(self.horizon-1)
        return np.concatenate((self.current_obs,self.controller.planned_action,[phase])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs,info=self.env.reset(seed=seed,options=options); self.hard_sampled=False
        if self.sampling_mode=='hard':
            x,y=self._sample_position(); self.env.cube_x=x;self.env.cube_y=y
            q=self.env.cube_qpos; self.env.data.qpos[q:q+7]=[x,y,.425,1,0,0,0]
            mujoco.mj_forward(self.env.model,self.env.data);self.env.initial_z=float(self.env.data.xpos[self.env.cube_id,2])
            self.env.previous_distance=self.env._distance_to_grasp();self.env.hold_steps=0;obs=self.env._observe()
        self.current_obs=obs.copy(); self.previous_distance=self.env._distance_to_grasp();self.previous_lift=0.
        self.bc_latencies=[];self.total_bc_latency=0.;self.bc_inference_calls=0
        import time
        start=time.perf_counter();self.controller.reset(obs);elapsed=time.perf_counter()-start
        self.bc_latencies.append(elapsed);self.total_bc_latency+=elapsed;self.bc_inference_calls+=1
        info=dict(info);info['hard_sampled']=self.hard_sampled
        return self._augmented_obs(),info

    def step(self, action):
        if self.current_obs is None: raise RuntimeError('Call reset() first')
        residual=np.clip(np.asarray(action,dtype=np.float32),-1.,1.)
        if residual.shape!=(3,):raise ValueError('Expected 3D SAC residual')
        bc_action=self.controller.planned_action
        executed_chunk_index=int(self.controller.chunk_index)
        final=bc_action.copy();final[:3]=np.clip(bc_action[:3]+residual*self.residual_meters/self.env.max_move,-1.,1.)
        final[3]=bc_action[3]
        commanded=residual*self.residual_meters
        applied=(final[:3]-bc_action[:3])*self.env.max_move
        obs,_,terminated,truncated,info=self.env.step(final);self.current_obs=obs.copy()
        dist=float(info['distance']);lift=float(info['lift_height'])
        task=-.01+(20. if info['is_success'] else (-5. if truncated else 0.))
        penalty=-self.residual_penalty_coefficient*float(np.sum(residual**2))
        dp=float(np.clip(self.previous_distance-dist,-.03,.03));lp=float(np.clip(lift-self.previous_lift,-.03,.03))
        shaping=(10.*dp+15.*lp) if self.reward_mode=='shaped' else 0.
        reward=task+penalty+shaping;self.previous_distance=dist;self.previous_lift=lift
        # Update controller only after executing the currently exposed action.
        import time
        start=time.perf_counter();replan=self.controller.advance(obs,terminated,truncated);elapsed=time.perf_counter()-start
        if replan:
            self.bc_latencies.append(elapsed);self.total_bc_latency+=elapsed;self.bc_inference_calls+=1
        info=dict(info);info.update(hard_sampled=self.hard_sampled,task_reward=task,
            residual_action_penalty=penalty,shaping_reward=shaping,distance_progress=dp,lift_progress=lp,
            residual_norm=float(np.linalg.norm(residual*self.residual_meters)),
            residual_action_norm=float(np.linalg.norm(residual)),action_saturated=bool(np.any(np.abs(residual)>=.95)),
            commanded_residual_xyz_m=commanded.copy(),applied_residual_xyz_m=applied.copy(),
            applied_residual_norm_m=float(np.linalg.norm(applied)),bc_action=bc_action.copy(),
            executed_action=final.copy(),chunk_index=executed_chunk_index,
            chunk_replanned=bool(replan),bc_inference_calls=self.bc_inference_calls)
        return self._augmented_obs(),float(reward),terminated,truncated,info

    def training_state_dict(self):
        b=self.env; spec=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(b.model,spec),dtype=np.float64);mujoco.mj_getState(b.model,b.data,state,spec)
        return {'integration_state':state,'ctrl':b.data.ctrl.copy(),'cube_x':b.cube_x,'cube_y':b.cube_y,
            'initial_z':b.initial_z,'base_ctrl':b.ctrl.copy(),'gripper_command':b.gripper_command,
            'base_previous_distance':b.previous_distance,'hold_steps':b.hold_steps,
            'base_rng':b.np_random.bit_generator.state,'current_obs':None if self.current_obs is None else self.current_obs.copy(),
            'previous_distance':self.previous_distance,'previous_lift':self.previous_lift,'hard_sampled':self.hard_sampled,
            'controller':self.controller.state_dict(),'bc_latencies':list(self.bc_latencies),
            'total_bc_latency':self.total_bc_latency,'bc_inference_calls':self.bc_inference_calls}

    def load_training_state_dict(self,s):
        b=self.env;spec=mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(b.model,b.data,s['integration_state'],spec);b.data.ctrl[:]=s['ctrl'];mujoco.mj_forward(b.model,b.data)
        b.cube_x=s['cube_x'];b.cube_y=s['cube_y'];b.initial_z=s['initial_z'];b.ctrl=s['base_ctrl'].copy()
        b.gripper_command=s['gripper_command'];b.previous_distance=s['base_previous_distance'];b.hold_steps=s['hold_steps']
        b.np_random.bit_generator.state=s['base_rng'];self.current_obs=None if s['current_obs'] is None else s['current_obs'].copy()
        self.previous_distance=s['previous_distance'];self.previous_lift=s['previous_lift'];self.hard_sampled=s['hard_sampled']
        self.controller.load_state_dict(s['controller']);self.bc_latencies=list(s['bc_latencies'])
        self.total_bc_latency=s['total_bc_latency'];self.bc_inference_calls=s['bc_inference_calls']
