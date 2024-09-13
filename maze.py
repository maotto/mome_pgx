from brax import envs
from brax import jumpy as jp

from qdax.environments import MazeWrapper

# choose in ["ant"]
ENV_NAME = "ant"
env = envs.create(env_name=ENV_NAME)
qd_env = MazeWrapper(env, ENV_NAME)

state = qd_env.reset(rng=jp.random_prngkey(seed=0))
for i in range(10):
    action = jp.zeros((qd_env.action_size,))
    state = qd_env.step(state, action)