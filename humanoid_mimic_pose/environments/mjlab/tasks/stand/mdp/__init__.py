from mjlab.envs.mdp import *

# Re-exported from the velocity task's mdp — these are not yet ported locally.
# Swap to local implementations in .rewards / .observations if you want to
# customize them for standing.
from mjlab.tasks.velocity.mdp import (
    angular_momentum_penalty as angular_momentum_penalty,
    body_angular_velocity_penalty as body_angular_velocity_penalty,
    foot_contact as foot_contact,
    foot_contact_forces as foot_contact_forces,
    upright as upright,
)

from .rewards import *
