import asyncio
import dataclasses
import inspect
import json
import logging
import random
import re
import typing

import jinja2
import pydantic
import tenacity
import tqdm

import llm

logger = logging.getLogger(__name__)
TEMPLATES = jinja2.Environment(
    loader=jinja2.FileSystemLoader("templates_v2"), undefined=jinja2.StrictUndefined
)

print(
    f"Using endpoint {llm.OPENAI_BASE_URL} with models {list(set((llm.DEFAULT_MODEL, llm.BIG_MODEL)))}"
    + (" and chain of thought" if llm.BIG_MODEL_CHAIN_OF_THOUGHT else "")
)


@dataclasses.dataclass
class IndHist:
    parents: typing.List[dict]
    prompt: str
    child: dict


DUPLICATE_PROMPTS = False
DEBUGGING = False  # Set to True to use dummy prompts instead of LLM for gridworld task
STATIC_POOLS = False  # Set to True to disable updating the mutation and recombination pools
CHAT_MONKEY = False  # use parallel sampling


@dataclasses.dataclass
class _FitIndividual:
    ind: dict
    fitness: float


class Solver:
    def __init__(
        self,
        task_str: str,
        fitness_function: typing.Callable,
        schema: typing.Optional[typing.Type[pydantic.BaseModel]] = None,
    ):
        self.task_str = task_str
        self.fitness_function = fitness_function
        self.signature = f"{fitness_function.__name__}{inspect.signature(fitness_function)}"
        self.hist_ppool_rec: typing.List[typing.List[str]] = []
        self.hist_ppool_mut: typing.List[typing.List[str]] = []
        self.ind_hist: typing.List[typing.List[IndHist]] = []
        self.ind_hist_lock = asyncio.Lock()
        if schema is None:
            self.schema: dict = {}
        else:
            self.schema = schema.model_json_schema()
        self.history: typing.List[typing.List[_FitIndividual]] = []

    async def safe_append(self, ind_hist: IndHist):
        async with self.ind_hist_lock:
            self.ind_hist[-1].append(ind_hist)

    async def solve(
        self,
        population_size=10,
        num_generations=10,
        pool_mut_len=3,
        pool_rec_len=3,
        rate_elite=0.1,
        rate_mut=0.4,
        rate_rec=0.4,
        log_file: typing.Optional[str] = None,
    ):
        if not self.schema:
            await self._create_schema()
        self._log(log_file)
        jobs = [self._create_genome(self.schema) for _ in range(population_size)]
        job_res = await asyncio.gather(*jobs)
        individs = [_FitIndividual(ind=ind, fitness=self.fitness_function(**ind)) for ind in job_res]
        self.history.append(sorted(individs, key=lambda x: x.fitness, reverse=True))
        self._log(log_file)
        await self._create_ppools(self.history[-1][0].ind, pool_mut_len, pool_rec_len)
        self._log(log_file)
        num_elites = int(len(self.history[-1]) * rate_elite)
        num_mut = int(len(self.history[-1]) * rate_mut)
        num_rec = int(len(self.history[-1]) * rate_rec)
        num_kept = len(self.history[-1]) - num_elites - num_mut - num_rec
        logger.info(
            f"Starting evolution with {num_elites} elites, {num_mut} mutations, {num_rec} recombinations, and {num_kept} kept"
        )
        for _ in (pbar := tqdm.tqdm(range(num_generations))):
            self.ind_hist.append([])
            await self._evolve(num_elites, num_mut, num_rec, num_kept)
            best_fitness = self.history[-1][0].fitness
            average_fitness = sum([ind.fitness for ind in self.history[-1]]) / len(self.history[-1])
            pbar.set_description(f"Best fitness: {best_fitness:.2f}, Average fitness: {average_fitness:.2f}")
            await self._update_ppols()
            self._log(log_file)
        return self.history[-1][0].ind

    def _log(self, log_file):
        if log_file:
            with open(log_file, "w") as f:
                json.dump(
                    {
                        "schema": self.schema,
                        "hist_ppool_mut": self.hist_ppool_mut,
                        "hist_ppool_rec": self.hist_ppool_rec,
                        "history": [[dataclasses.asdict(ind) for ind in epoch] for epoch in self.history],
                        "ind_hist": [[dataclasses.asdict(ind) for ind in epoch] for epoch in self.ind_hist],
                    },
                    f,
                )

    async def _create_ppools(self, example: dict, pool_mut_len: int, pool_rec_len: int):
        assert pool_mut_len > 0, "Mutation pool size must be greater than 0"
        assert pool_rec_len > 0, "Recombination pool size must be greater than 0"
        if DEBUGGING:
            self.hist_ppool_mut.append(["Swap some elements"] * pool_mut_len)
            self.hist_ppool_rec.append(["Swap some elements"] * pool_rec_len)
            return

        async def _call_p(c) -> str:
            p = TEMPLATES.get_template(c).render(task_str=self.task_str, example=example, schema=self.schema)
            return await llm.chat(  # type: ignore
                system_prompt=p if DUPLICATE_PROMPTS else None,
                user_prompt=p,
                llm_args={"temperature": 1},
                model=llm.BIG_MODEL,
            )

        jobs_mut = [_call_p("create_mutp.md") for _ in range(pool_mut_len)]
        jobs_rec = [_call_p("create_recp.md") for _ in range(pool_rec_len)]
        new_mut, new_rec = await asyncio.gather(asyncio.gather(*jobs_mut), asyncio.gather(*jobs_rec))
        self.hist_ppool_mut.append(new_mut)  # type: ignore
        self.hist_ppool_rec.append(new_rec)  # type: ignore

    async def _update_ppols(self):
        if STATIC_POOLS:
            return
        if len(self.history) < 2:
            self.hist_ppool_rec.append(self.hist_ppool_rec[-1])
            self.hist_ppool_mut.append(self.hist_ppool_mut[-1])
            return
        new_individuals = self.history[-1][:4]
        old_individuals = self.history[-2][:4]

        async def _call_p(c, prompt) -> str:
            p = TEMPLATES.get_template(c).render(
                task_str=self.task_str,
                old_individuals=old_individuals,
                new_individuals=new_individuals,
                example=new_individuals[0].ind,
                old_prompt=prompt,
            )
            return await llm.chat(  # type: ignore
                system_prompt=p if DUPLICATE_PROMPTS else None,
                user_prompt=p,
                llm_args={"temperature": 1},
                model=llm.BIG_MODEL,
            )

        jobs_mut = [_call_p("improve_mut.md", prompt) for prompt in self.hist_ppool_mut[-1]]
        jobs_rec = [_call_p("improve_rec.md", prompt) for prompt in self.hist_ppool_rec[-1]]

        new_mut, new_rec = await asyncio.gather(asyncio.gather(*jobs_mut), asyncio.gather(*jobs_rec))
        self.hist_ppool_mut.append(new_mut)  # type: ignore
        self.hist_ppool_rec.append(new_rec)  # type: ignore

    async def _evolve(self, num_elites: int, num_mut: int, num_rec: int, num_kept: int):
        pop_avg = sum([ind.fitness for ind in self.history[-1]]) / len(self.history[-1])
        new_pop = []
        new_pop.extend(self.history[-1][:num_elites])
        prompts_mut = random.choices(self.hist_ppool_mut[-1], k=num_mut)
        prompts_rec = random.choices(self.hist_ppool_rec[-1], k=num_rec)

        jobs = []
        for individual, prompt in zip(random.sample(self.history[-1], num_mut), prompts_mut):
            jobs.append(self._mutate(individual, prompt, pop_avg))

        for individuals, prompt in zip(
            [random.sample(self.history[-1], 2) for _ in range(num_rec)], prompts_rec
        ):
            jobs.append(self._recombine(individuals[0], individuals[1], prompt, pop_avg))

        job_res = await asyncio.gather(*jobs)
        new_pop.extend([_FitIndividual(ind=ind, fitness=self.fitness_function(**ind)) for ind in job_res])

        for _ in range(num_kept):
            new_pop.append(random.choice(self.history[-1]))

        new_pop = sorted(new_pop, key=lambda x: x.fitness, reverse=True)
        self.history.append(new_pop)

    async def _mutate(self, individual: _FitIndividual, mutation_prompt: str, pop_average: float):
        prompt = TEMPLATES.get_template("exec_mut.md").render(
            task_str=self.task_str,
            fitness=individual.fitness,
            individual=individual.ind,
            pop_average=pop_average,
            mutation_prompt=mutation_prompt,
        )
        genome_str = await llm.chat(
            system_prompt=prompt if DUPLICATE_PROMPTS else None,
            user_prompt=prompt,
            schema=self.schema,
            llm_args={"temperature": 1},
        )
        assert isinstance(genome_str, str)

        d = llm._str2dict(genome_str)
        await self.safe_append(IndHist([individual.ind], prompt, d))
        return d

    async def _recombine(
        self,
        individual1: _FitIndividual,
        individual2: _FitIndividual,
        recombination_prompt: str,
        pop_average: float,
    ):
        prompt = TEMPLATES.get_template("exec_rec.md").render(
            task_str=self.task_str,
            fitness1=individual1.fitness,
            fitness2=individual2.fitness,
            individual1=individual1.ind,
            individual2=individual2.ind,
            pop_average=pop_average,
            recombination_prompt=recombination_prompt,
        )
        genome_str = await llm.chat(
            system_prompt=prompt if DUPLICATE_PROMPTS else None,
            user_prompt=prompt,
            schema=self.schema,
            llm_args={"temperature": 1},
        )
        assert isinstance(genome_str, str)
        d = llm._str2dict(genome_str)
        await self.safe_append(IndHist([individual1.ind, individual2.ind], prompt, d))
        return d

    async def _create_genome(self, schema: dict) -> dict:
        if DEBUGGING:
            return {"path": random.choices(["up", "down", "left", "right"], k=20)}

        prompt = TEMPLATES.get_template("create_genome.md").render(
            task_str=self.task_str, signature=self.signature, schema=schema
        )
        genome_str = await llm.chat(
            system_prompt=prompt if DUPLICATE_PROMPTS else None,
            user_prompt=prompt,
            schema=schema,
            llm_args={"temperature": 1},
        )
        if isinstance(genome_str, str):
            return llm._str2dict(genome_str)
        else:
            raise llm.GenerationError("Model returned invalid", genome_str)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3), retry=tenacity.retry_if_exception_type(llm.GenerationError)
    )
    async def _create_schema(self):
        if DEBUGGING:
            self.schema = {
                "properties": {
                    "path": {
                        "items": {"enum": ["up", "down", "left", "right"], "type": "string"},
                        "title": "Path",
                        "minItems": 15,
                        "maxItems": 50,
                        "type": "array",
                    }
                },
                "required": ["path"],
                "title": "Model",
                "type": "object",
            }
            return

        prompt = TEMPLATES.get_template("create_json.md").render(
            task_str=self.task_str, signature=self.signature
        )

        if CHAT_MONKEY:
            resp = await llm.chat_monkey(
                system_prompt=prompt if DUPLICATE_PROMPTS else None,
                user_prompt=prompt,
                verifier_system_prompt=TEMPLATES.get_template("monkey_genome.md").render(
                    signature=self.signature
                ),
            )
        else:
            resp = await llm.chat(
                system_prompt=prompt if DUPLICATE_PROMPTS else None,
                user_prompt=prompt,
                llm_args={"temperature": 1},
                model=llm.BIG_MODEL,
            )

        assert isinstance(resp, str)
        match = re.search(r"```(json|JSON|python)\n(.*?)\n```", resp, re.DOTALL)
        if match:
            schema_str = match.group(0)
        else:
            logger.info(f"Could not find schema in response: {resp}, prompt was {prompt.__repr__()}")
            raise llm.GenerationError("Could not find schema in response")
        try:
            schema = llm._str2dict(schema_str)
        except Exception as e:
            logger.info(
                f"Could not parse schema into dict: {schema_str}, response was {resp}, match was {match.group(1)}, error was {e}"
            )
            raise llm.GenerationError("Could not parse schema")
        self.schema = schema
