import jax
import jax.numpy as jnp

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.types import Descriptor
from jax.experimental import host_callback as hcb


def get_final_xy_position(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Compute final xy positon.

    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
    descriptors = jax.vmap(lambda x, y: x[y])(data.state_desc, last_index)

    # remove the dim coming from the trajectory
    return descriptors.squeeze(axis=1)


def get_feet_contact_proportion(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    """Compute feet contact time proportion.

    This function suppose that state descriptor is the feet contact, as it
    just computes the mean of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    descriptors = jnp.sum(data.state_desc * (1.0 - mask), axis=1)
    descriptors = descriptors / jnp.sum(1.0 - mask, axis=1)

    return descriptors

def get_joint_workspace_ratio(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    kick_mask = jnp.any(data.state_desc != 0, axis=-1).astype(int)
    kick_mask = jnp.expand_dims(kick_mask, axis=-1)

    new_mask = (1 - mask) * kick_mask
    descriptors = jnp.sum(data.state_desc * new_mask, axis=1)
    

    descriptors = descriptors / jnp.sum(new_mask, axis=1)
    # def shifted_sigmoid(x, shift=8.0):
    #     return 1 / (1 + jnp.exp(-0.4*(x - shift)))

    # descriptors = shifted_sigmoid(descriptors)    
    return descriptors

def get_exp_and_ratio(data: QDTransition, mask: jnp.ndarray) -> Descriptor:
    mask = jnp.expand_dims(mask, axis=-1)
    print(f"{mask.shape=}")
    kick_mask = jnp.any(data.state_desc != 0, axis=-1).astype(int)
    kick_mask = jnp.expand_dims(kick_mask, axis=-1)
    print(f"{kick_mask.shape=}")

    new_mask = (1 - mask) * kick_mask
    #print(f"{new_mask.shape=}")
    descriptors = jnp.sum(data.state_desc * new_mask, axis=1)
    #print(f"{descriptors.shape=}")
    contact_timesteps = descriptors[:, 1]
    total_ratios = descriptors[:, 0]
    # hcb.id_print(descriptors, what="Descriptors before normalization")

    def shifted_sigmoid(x, shift=50.0):
        return 1 / (1 + jnp.exp(-0.1*(x - shift)))

    ts_sigmoid = shifted_sigmoid(contact_timesteps)
    total_ratios = jnp.where(contact_timesteps == 0, 0.0, total_ratios)
    contact_timesteps = jnp.where(contact_timesteps == 0, 1.0, contact_timesteps)
    normalized_ratios = total_ratios / contact_timesteps

    descriptors = jnp.stack((ts_sigmoid, normalized_ratios), axis=-1)
    # hcb.id_print(descriptors, what="Final descriptors")
    #print(f"{descriptors.shape=}")
    return descriptors