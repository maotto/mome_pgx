from typing import Any, List, Optional, Sequence, Tuple

import jax.numpy as jnp
from sklearn.decomposition import PCA
from brax import jumpy as jp
from brax.envs import Env, State, Wrapper
from brax.physics import config_pb2
from brax.physics.base import QP, Info
from brax.physics.system import System
import jax
from jax.experimental import host_callback as hcb


from qdax.environments.base_wrappers import QDEnv

FEET_NAMES = {
    "ant": ["$ Body 4", "$ Body 7", "$ Body 10", "$ Body 13"],
    "halfcheetah": ["ffoot", "bfoot"],
    "walker2d": ["foot", "foot_left"],
    "hopper": ["foot"],
    "humanoid": ["left_shin", "right_shin"],
    "kicker": ["$ Body 4"],
    "jumper": ["$ Body 4", "$ Body 10"],
    "kicker_exp": ["$ Body 4"]
}


class QDSystem(System):
    """Inheritance of brax physic system.

    Work precisely the same but store some information from the physical
    simulation in the aux_info attribute.

    This is used in FeetContactWrapper to get the feet contact of the
    robot with the ground.
    """

    def __init__(
        self, config: config_pb2.Config, resource_paths: Optional[Sequence[str]] = None
    ):
        super().__init__(config, resource_paths=resource_paths)
        self.aux_info = None

    def step(self, qp: QP, act: jp.ndarray) -> Tuple[QP, Info]:
        qp, info = super().step(qp, act)
        self.aux_info = info
        return qp, info


class FeetContactWrapper(QDEnv):
    """Wraps gym environments to add the feet contact data.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the feet_contact booleans in
    the information dictionary of the Brax.state.

    The only supported envs at the moment are among the classic
    locomotion envs : Walker2D, Hopper, Ant, Bullet.

    New locomotions envs can easily be added by adding the config name
    of the feet of the corresponding environment in the FEET_NAME dictionary.

    Example :

        from brax import envs
        from brax import jumpy as jp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = FeetContactWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jp.random_prngkey(seed=0))
        for i in range(10):
            action = jp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)

            # retrieve feet contact
            feet_contact = state.info["state_descriptor"]

            # do whatever you want with feet_contact
            print(f"Feet contact : {feet_contact}")


    """

    def __init__(self, env: Env, env_name: str):

        if env_name not in FEET_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(config=None)

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self.env.sys = QDSystem(self.env.sys.config)

        self._feet_contact_idx = jp.array(
            [env.sys.body.index.get(name) for name in FEET_NAMES[env_name]]
        )

    @property
    def state_descriptor_length(self) -> int:
        return self.behavior_descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return "feet_contact"

    @property
    def state_descriptor_limits(self) -> Tuple[List, List]:
        return self.behavior_descriptor_limits

    @property
    def behavior_descriptor_length(self) -> int:
        return len(self._feet_contact_idx)

    @property
    def behavior_descriptor_limits(self) -> Tuple[List, List]:
        bd_length = self.behavior_descriptor_length
        return (jnp.zeros((bd_length,)), jnp.ones((bd_length,)))

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = self._get_feet_contact(
            self.env.sys.info(state.qp)
        )
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = self._get_feet_contact(self.env.sys.aux_info)
        return state

    def _get_feet_contact(self, info: Info) -> jp.ndarray:
        contacts = info.contact.vel
        return jp.any(contacts[self._feet_contact_idx], axis=1).astype(jp.float32)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


# name of the center of gravity
COG_NAMES = {
    "ant": "$ Torso",
    "halfcheetah": "torso",
    "walker2d": "torso",
    "hopper": "torso",
    "humanoid": "torso",
}


class XYPositionWrapper(QDEnv):
    """Wraps gym environments to add the position data.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply add the actual position in
    the information dictionary of the Brax.state.

    One can also add values to clip the state descriptors.

    The only supported envs at the moment are among the classic
    locomotion envs : Ant, Humanoid.

    New locomotions envs can easily be added by adding the config name
    of the feet of the corresponding environment in the STATE_POSITION
    dictionary.

    RMQ: this can be used with Hopper, Walker2d, Halfcheetah but it makes
    less sens as those are limited to one direction.

    Example :

        from brax import envs
        from brax import jumpy as jp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah", "humanoid"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = XYPositionWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jp.random_prngkey(seed=0))
        for i in range(10):
            action = jp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)

            # retrieve feet contact
            xy_position = state.info["xy_position"]

            # do whatever you want with xy_position
            print(f"xy position : {xy_position}")


    """

    def __init__(
        self,
        env: Env,
        env_name: str,
        minval: Optional[List[float]] = None,
        maxval: Optional[List[float]] = None,
    ):
        if env_name not in COG_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(config=None)

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self._cog_idx = self.env.sys.body.index[COG_NAMES[env_name]]
        else:
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        if minval is None:
            minval = jnp.ones((2,)) * (-jnp.inf)

        if maxval is None:
            maxval = jnp.ones((2,)) * jnp.inf

        if len(minval) == 2 and len(maxval) == 2:
            self._minval = jnp.array(minval)
            self._maxval = jnp.array(maxval)
        else:
            raise NotImplementedError(
                "Please make sure to give two values for each limits."
            )

    @property
    def state_descriptor_length(self) -> int:
        return 2

    @property
    def state_descriptor_name(self) -> str:
        return "xy_position"

    @property
    def state_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self._minval, self._maxval

    @property
    def behavior_descriptor_length(self) -> int:
        return self.state_descriptor_length

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.state_descriptor_limits

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = jnp.clip(
            state.qp.pos[self._cog_idx][:2], a_min=self._minval, a_max=self._maxval
        )
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        # get xy position of the center of gravity
        state.info["state_descriptor"] = jnp.clip(
            state.qp.pos[self._cog_idx][:2], a_min=self._minval, a_max=self._maxval
        )
        return state

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


# name of the forward/velocity reward
FORWARD_REWARD_NAMES = {
    "ant": "reward_forward",
    "halfcheetah": "reward_run",
    "walker2d": "reward_forward",
    "hopper": "reward_forward",
    "humanoid": "forward_reward",
    "jumper": "reward_forward"
}


class NoForwardRewardWrapper(Wrapper):
    """Wraps gym environments to remove forward reward.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply remove the forward speed term
    of the reward.

    Example :

        from brax import envs
        from brax import jumpy as jp

        # choose in ["ant", "walker2d", "hopper", "halfcheetah", "humanoid"]
        ENV_NAME = "ant"
        env = envs.create(env_name=ENV_NAME)
        qd_env = NoForwardRewardWrapper(env, ENV_NAME)

        state = qd_env.reset(rng=jp.random_prngkey(seed=0))
        for i in range(10):
            action = jp.zeros((qd_env.action_size,))
            state = qd_env.step(state, action)
    """

    def __init__(self, env: Env, env_name: str) -> None:
        if env_name not in FORWARD_REWARD_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")
        super().__init__(env)
        self._env_name = env_name
        self._fd_reward_field = FORWARD_REWARD_NAMES[env_name]

    @property
    def name(self) -> str:
        return self._env_name

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        # update the reward (remove forward_reward)
        new_reward = state.reward - state.metrics[self._fd_reward_field]
        return state.replace(reward=new_reward)  # type: ignore


MINIMISE_ENERGY_REWARD_NAMES = {
    "ant": "reward_ctrl",
    "halfcheetah": "reward_ctrl",
    "walker2d": "reward_ctrl",
    "hopper": "reward_ctrl",
    "humanoid": "reward_quadctrl",
    "kicker": "reward_ctrl",
    "jumper": "reward_ctrl",
    "kicker_exp": "reward_ctrl",
}

class MultiObjectiveRewardWrapper(Wrapper):
    """Wraps gym environments and replaces reward with multi-objective rewards.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply use forward reward and negative energy as rewards.
    """

    def __init__(self, env: Env, env_name: str) -> None:
        if env_name not in (FORWARD_REWARD_NAMES.keys() and MINIMISE_ENERGY_REWARD_NAMES.keys()):
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")
        super().__init__(env)
        self._env_name = env_name
        self._fd_reward_field = FORWARD_REWARD_NAMES[env_name]
        self._minimise_energy_reward_field = MINIMISE_ENERGY_REWARD_NAMES[env_name]

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        new_reward = jp.zeros((2,))
        return state.replace(reward=new_reward)

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        new_reward = jnp.concatenate(
            (jp.array((state.metrics[self._fd_reward_field],)), 
            jp.array((state.metrics[self._minimise_energy_reward_field],))), 
            axis=-1
        )
        return state.replace(reward=new_reward)  
JOINTED_BODIES_NAMES = {
    "kicker": ["hip_1", "ankle_1"],
    "kicker_exp": ["hip_1", "ankle_1"]
}


class JointRangeWrapper(QDEnv):
    """Wraps gym environments to add the used joint range data.

    """

    def __init__(self, env: Env, env_name: str):

        if env_name not in JOINTED_BODIES_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(config=None)

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self.env.sys = QDSystem(self.env.sys.config)

        self._joints_idx = jp.array(
            [env.sys.joints[0].index.get(name) for name in JOINTED_BODIES_NAMES[env_name]]
        )
        self._feet_contact_idx = jp.array(
            [env.sys.body.index.get(name) for name in FEET_NAMES[env_name]]
        )


    @property
    def state_descriptor_length(self) -> int:
        return self.behavior_descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return "joint_range"

    @property
    def state_descriptor_limits(self) -> Tuple[List, List]:
        return self.behavior_descriptor_limits

    @property
    def behavior_descriptor_length(self) -> int:
        return len(self._joints_idx)

    @property
    def behavior_descriptor_limits(self) -> Tuple[List, List]:
        bd_length = self.behavior_descriptor_length
        return (jnp.zeros((bd_length,)), jnp.ones((bd_length,)))

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = jnp.clip(self._get_joint_angles_ratio(state), 0, 1) * self._get_feet_contact(
            self.env.sys.info(state.qp)
        )
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = jnp.clip(self._get_joint_angles_ratio(state), 0, 1) * self._get_feet_contact(self.env.sys.aux_info)
        return state

    def _get_joint_angles_ratio(self, state: State) -> jp.ndarray:
        
        joint_angles, joint_vel = self.env.sys.joints[0].angle_vel(state.qp)

        relevant_angles = joint_angles[self._joints_idx]
        joint_limits = self.env.sys.joints[0].limit[self._joints_idx]
        min_limits = joint_limits[:, 0, 0]
        max_limits = joint_limits[:, 0, 1]

        normalized_angles = (relevant_angles - min_limits) / (max_limits - min_limits)

        return normalized_angles
    
    def _get_feet_contact(self, info: Info) -> float:
        contacts = info.contact.vel
        return jp.any(contacts[self._feet_contact_idx], axis=1).astype(jp.float32)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

class ExplorationWrapper(QDEnv):
    """Wraps gym environments to add the used joint range data.

    """

    def __init__(self, env: Env, env_name: str):

        if env_name not in JOINTED_BODIES_NAMES.keys():
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(config=None)

        self.env = env
        self._env_name = env_name
        if hasattr(self.env, "sys"):
            self.env.sys = QDSystem(self.env.sys.config)

        self._joints_idx = jp.array(
            [env.sys.joints[0].index.get(name) for name in JOINTED_BODIES_NAMES[env_name]]
        )
        self._feet_contact_idx = jp.array(
            [env.sys.body.index.get(name) for name in FEET_NAMES[env_name]]
        )
        self.object_pos = jp.array([0.7, 0.7, 0.18])


    @property
    def state_descriptor_length(self) -> int:
        return self.behavior_descriptor_length

    @property
    def state_descriptor_name(self) -> str:
        return "joint_range"

    @property
    def state_descriptor_limits(self) -> Tuple[List, List]:
        return self.behavior_descriptor_limits

    @property
    def behavior_descriptor_length(self) -> int:
        return len(self._joints_idx)

    @property
    def behavior_descriptor_limits(self) -> Tuple[List, List]:
        bd_length = self.behavior_descriptor_length
        return (jnp.zeros((bd_length,)), jnp.ones((bd_length,)))

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = jp.array([self._get_joint_angles_ratio(state), 1.0]) * self._get_feet_contact(self.env.sys.info(state.qp))
        # hcb.id_print(state.info["state_descriptor"], what="Reset: behavior_descriptor")
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = jp.array([self._get_joint_angles_ratio(state), 1.0]) * self._get_feet_contact(self.env.sys.aux_info)
        # hcb.id_print(state.info["state_descriptor"], what="Step: behavior_descriptor")
        return state


    def _get_joint_angles_ratio(self, state: State) -> float:
        
        joint_angles, joint_vel = self.env.sys.joints[0].angle_vel(state.qp)

        relevant_angles = joint_angles[self._joints_idx]
        joint_limits = self.env.sys.joints[0].limit[self._joints_idx]
        min_limits = joint_limits[:, 0, 0]
        max_limits = joint_limits[:, 0, 1]

        normalized_angles = (relevant_angles - min_limits) / (max_limits - min_limits)
        sigmoid = self.shifted_sigmoid(normalized_angles[0]+normalized_angles[1])
        # mean = jnp.mean(normalized_angles) 
        # mean = jnp.clip(mean, 0.0, 1.0)

        return sigmoid
    def shifted_sigmoid(self, x, shift=1.0):
            return 1 / (1 + jnp.exp(-3*(x - shift)))
    
    def _get_feet_contact(self, info: Info) -> float:
        contacts = info.contact.vel
        return jp.any(contacts[self._feet_contact_idx], axis=1).astype(jp.float32)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)



ACCURACY_REWARD_NAMES = {
    "kicker": "reward_dist",
    "kicker_exp": "reward_dist",
}
class MORewardWrapper(Wrapper):
    """Wraps gym environments and replaces reward with multi-objective rewards.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply use forward reward and negative energy as rewards.
    """

    def __init__(self, env: Env, env_name: str) -> None:
        if env_name not in (ACCURACY_REWARD_NAMES.keys() and MINIMISE_ENERGY_REWARD_NAMES.keys()):
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")
        super().__init__(env)
        self._env_name = env_name
        self._acc_reward_field = ACCURACY_REWARD_NAMES[env_name]
        self._minimise_energy_reward_field = MINIMISE_ENERGY_REWARD_NAMES[env_name]

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        new_reward = jp.zeros((2,))
        return state.replace(reward=new_reward)

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        new_reward = jnp.concatenate(
            (jp.array((state.metrics[self._acc_reward_field],)), 
            jp.array((state.metrics[self._minimise_energy_reward_field],))), 
            axis=-1
        )
        return state.replace(reward=new_reward)  

EXPLORATION_REWARD_NAMES = {
    "kicker": "reward_near",
}
class TriORewardWrapper(Wrapper):
    """Wraps gym environments and replaces reward with multi-objective rewards.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply use forward reward and negative energy as rewards.
    """

    def __init__(self, env: Env, env_name: str) -> None:
        if env_name not in (ACCURACY_REWARD_NAMES.keys() and MINIMISE_ENERGY_REWARD_NAMES.keys() and EXPLORATION_REWARD_NAMES.keys()):
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")
        super().__init__(env)
        self._env_name = env_name
        self._acc_reward_field = ACCURACY_REWARD_NAMES[env_name]
        self._minimise_energy_reward_field = MINIMISE_ENERGY_REWARD_NAMES[env_name]
        self._exp_reward_field = EXPLORATION_REWARD_NAMES[env_name]

    @property
    def name(self) -> str:
        return self._env_name

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        new_reward = jp.zeros((3,))
        return state.replace(reward=new_reward)

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        new_reward = jnp.concatenate(
            (jp.array((state.metrics[self._acc_reward_field],)), 
            jp.array((state.metrics[self._minimise_energy_reward_field],)),
            jp.array((state.metrics[self._exp_reward_field],))), 
            axis=-1
        )
        return state.replace(reward=new_reward)  