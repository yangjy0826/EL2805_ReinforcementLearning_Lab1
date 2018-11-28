import numpy as np
from random import randint


# actions
up = 0
down = 1
left = 2
right = 3
stand_still = 4
# maze size
MazeX = 4
MazeY = 4

# Create the maze, and position of bank
bank = np.zeros((MazeX, MazeY), dtype=np.bool)  # Type: True or false
bank[1, 1] = True


def limit_coordinates(coord):  # Limit the movement inside 4 by 4
    coord[0] = min(coord[0], MazeX - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], MazeY - 1)
    coord[1] = max(coord[1], 0)
    return coord


def calculate_transition(current_state, robber_movement):
    robber_new_position = np.array(current_state[0:1]) + np.array(robber_movement)
    robber_new_position = limit_coordinates(robber_new_position)  # Limit the movement inside 4 by 4
    i = randint(0, 3)
    if i == 0:
        police_movement = [0, -1]  # up
    elif i == 1:
        police_movement = [0, 1]  # down
    elif i == 2:
        police_movement = [-1, 0]  # left
    else:
        police_movement = [1, 0]  # right
    police_new_position = np.array(current_state[2:3]) + np.array(police_movement)
    police_new_position = limit_coordinates(police_new_position)  # Limit the movement inside 4 by 4
    new_position = np.append(robber_new_position, police_new_position)
    if robber_new_position[0] == police_new_position[0] and robber_new_position[1] == police_new_position[1]:
        reward = -10
    elif bank[tuple(robber_new_position)]:
        reward = 1
    else:
        reward = 0
    return new_position, reward


nS = MazeX*MazeY*MazeX*MazeY  # No of states in the state space
nA = 5  # No of actions for robber

R = {}  # Robber
Position = []
for s in range(nS):
    # np.unravel_index: converts a flat index or array of flat indices into a tuple of coordinate arrays:
    position = np.unravel_index(s, (MazeX, MazeY, MazeX, MazeY))
    Position.append(position)
    R[s] = {a: [] for a in range(nA)}  # Robber
    R[s][up] = calculate_transition(position, [0, -1])
    R[s][down] = calculate_transition(position, [0, 1])
    R[s][left] = calculate_transition(position, [-1, 0])
    R[s][right] = calculate_transition(position, [1, 0])
    R[s][stand_still] = calculate_transition(position, [0, 0])
# print(R[s][right])

lamb = 0.8  # discount factor
state_init = (0, 0, 3, 3)  # We always start from this state


