"""
Run LLMs on the Gridworld task.
"""

import asyncio
import typing
from typing import List, Literal
import llm
from pydantic import BaseModel, Field
import time

import solver
import utils

utils.setup_logging()

solver.STATIC_POOLS = False
llm.USE_LLAMA_CPP_COMPATIBLE = False


class Model(BaseModel):
    path: List[Literal["up", "down", "left", "right"]] = Field(..., min_length=10, max_length=30)


def fitness_function(path: typing.List[typing.Literal["up", "down", "left", "right"]]):
    for action in path:
        if action not in ["up", "down", "left", "right"]:
            return -100
    op = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    maze_structure = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    start, end = (1, 1), (8, 8)
    current = start
    reward = 0
    for action in path:
        next_position = (current[0] + op[action][0], current[1] + op[action][1])
        reward -= 0.1
        if maze_structure[next_position[0]][next_position[1]] == 0:
            current = next_position
        if current == end:
            reward += 50
            break
    if current != end:
        reward -= 50
    reward -= abs(current[0] - end[0]) + abs(current[1] - end[1])
    return reward


task_str = """
**Gridworld Task Description:**

In the Gridworld task, an agent navigates a 10 times 10 grid using four possible actions: up, down, left, right. The agent begins in the top-left corner of the grid, and its objective is to reach the bottom-right corner while adhering to the following rules and conditions:

1. **Movement Constraints:**  
   - The agent can move only to adjacent cells that are not blocked.
   - Any attempt to perform an invalid action (e.g., choosing an action other than the four specified) incurs a **penalty of -100**.
   - The agent cannot do more than 50 steps. The optimal solution uses 15 steps.

2. **Rewards System:**  
   - **Goal Reward:** Upon successfully reaching the bottom-right corner, the agent receives a **reward of 50**.
   - **Non-Goal Penalty:** When the agent fails to reach the goal, it incurs a **penalty of -50**.
   - **Step Penalty:** Each step taken incurs a **penalty of -0.1**, encouraging the agent to minimize the number of steps.
   - The agent receives a penalty of -1 for each unit distance from the goal. So, if the agent is 2 units up and 3 units left from the goal, it receives a penalty of -5.

3. **Objective:**  
   The primary goal is to navigate the grid from the starting position to the goal in as few steps as possible, maximizing the cumulative reward. However, solutions with less then 15 steps are considered invalid.

This task emphasizes efficient navigation, strategic decision-making, and avoiding unnecessary penalties.
"""

sol = solver.Solver(task_str, fitness_function, Model)
job = sol.solve(
    log_file=f"grid_dynamicpool{time.time()}.json",
    num_generations=10,
    population_size=20,
    rate_mut=0.45,
    rate_elite=0.1,
    rate_rec=0.45,
    pool_mut_len=4,
    pool_rec_len=4,
)

print(asyncio.run(job))
