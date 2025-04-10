# Code for the paper "NaLaGA"

## Paper: A General Genetic Algorithm Using Natural Language Evolutionary Operators
### Abstract

> By employing large language models (LLMs) we build a general genetic algorithm, i.e., a genetic algorithm (GA) that can solve various domains without any changes to its algorithmic components. Our approach requires only a problem description in natural language and a black-box fitness function and can then handle any type of data via natural-language-based evolutionary operators that call an LLM to compute their application. The relevant prompts for the operators can be human-designed or self-optimized with similar performance results. Compared to the only other generalist GA approach, i.e., asking an LLM to write a new specific GA, our natural-language-based genetic algorithm (NaLaGA) offers not only a better class of safety (since no LLM-generated code is executed by NaLaGA) but also greatly improved results in the two example domains Schwefel and grid world maze.


### Citation
Until the paper is published, please cite the preprint:
```bibtex
@inproceedings{stenzel2025genetic,
  title     = {A General Genetic Algorithm Using Natural Language Evolutionary Operators},
  author    = {Gerhard Stenzel and Sarah Gerner and Michael Kölle and Maximilian Zorn and Thomas Gabor},
  year      = {2025}
}
``` 

## Code

### Setup
Using [uv](https://docs.astral.sh/uv/getting-started/installation/): `uv sync`

### Configure
Rename `DEMO.env` to `.env`. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

### Flags
```python
solver.STATIC_POOLS = False # set to True to avoid updating the prompt pools
solver.CHAT_MONKEY = False # set to True to generate multiple solutions, and use an LLM to rank it to find the best.
solver.DUPLICATE_PROMPTS = False # set to True to set user prompts as same as system prompts
llm.BIG_MODEL = "llama3.2:3b" # use another model then llm.DEFAULT_MODEL
llm.BIG_MODEL_CHAIN_OF_THOUGHT = False # set to True if big model uses CoT (like QwQ or cogito)
llm.DISABLE_ALL_SCHEMA = False # set to True to disable all JSON constraints
```

### Run
`uv run run_grid.py` and `uv run run_simpleschwefel.py`

