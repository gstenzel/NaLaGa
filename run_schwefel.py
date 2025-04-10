import math
import asyncio
import solver
import utils
import pydantic

utils.setup_logging()


class Model(pydantic.BaseModel):
    a: float
    b: float
    c: float
    d: float


def fitness_function(a: float, b: float, c: float, d: float) -> float:
    x = [a, b, c, d]
    n = len(x)
    return -(418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x))


task_str = """
**Unknown Function Optimization**

In this task, you have to optimize a function with four parameters `a`, `b`, `c`, and `d`. Higher values of the resulting function are better. We do not know the exact function, but we know it contains some trigonometric operations. Your task is to find the combination of parameters that maximizes the function value.

**Rules and Conditions:**
1. You have to provide values for all four parameters.
2. You only have one function evaluation per generation.
3. The value range is between -500.0 and 500.0.

**Objective:**
Find the optimal combination of `a`, `b`, `c`, and `d` that maximize the function value.

"""

sol = solver.Solver(
    task_str,
    fitness_function,
    # schema=Model
)
job = sol.solve(
    log_file="DONOTSYNC_schwefel.json",
    num_generations=20,
    population_size=10,
)


print(asyncio.run(job))
