import carlo
from .state import State
from .action import Action

def get_reward(state: State, action: Action, world: carlo.World):
	# r = r(s, a)
    # - large positive reward for reaching the goal
    # - large negative reward for colliding or going off the road
    # - small negative reward for every tick we have not reached the goal
    r = 0
    # first check if we hit the goal (because the second condition here would
    # also be true if we hit the goal)
    if state.car.collidesWith(state.goal_pos):
        r = r + 100000
    elif world.collision_exists(state.car):
        # print(state.car.center)
        r = r - 10000
    else:
        r = r - state.car.distanceTo(state.goal_pos)

    return r
