import carlo
from .state import State
from .action import Action
import numpy as np

def get_reward(state: State, action: Action, world: carlo.World):
	# r = r(s, a)
    # - large positive reward for reaching the goal
    # - large negative reward for colliding or going off the road
    # - small negative reward for every tick we have not reached the goal
    r = 0
    # first check if we hit the goal (because the second condition here would
    # also be true if we hit the goal)
    if state.car.collidesWith(state.goal_pos):
        r = r + 1000000
    elif world.collision_exists(state.car):
        r = r - 100000
    elif world.is_out_of_world(state.car):
        r = r - 100000
    else:
        goal = state.goal_pos.center
        car = state.car.center
        dist = np.sqrt((goal.x-car.x)**2 + (goal.y-car.y)**2)
        if dist > 40:
            r = r - 1 - dist/12
        else:
            r = r + 40 - dist

    return r
