import ast
import asyncio
import logging
import os
import re
import string
import typing

import dotenv
import openai
from openai.types.model import Model
import pydantic

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") or "gpt-3.5-turbo"
BIG_MODEL = os.getenv("BIG_MODEL") or DEFAULT_MODEL
BIG_MODEL_CHAIN_OF_THOUGHT = os.getenv("BIG_MODEL_CHAIN_OF_THOUGHT") in ["true", "True", "1"]
DISABLE_ALL_SCHEMA = False

CLIENT = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
logger = logging.getLogger(__name__)

USE_LLAMA_CPP_COMPATIBLE = False


class GenerationError(Exception):
    pass


def _str2dict(s: str) -> dict:
    try:
        parsed = ast.literal_eval(
            s.strip("`")
            .strip("json")
            .strip("JSON")
            .replace("python", "")
            .replace("\n", " ")
            .replace("\t", " ")
        )
        if USE_LLAMA_CPP_COMPATIBLE:
            return parsed["properties"]
        return parsed
    except Exception as e:
        raise GenerationError(f"Error converting string to dict: {e}, {s.__repr__()}")


async def chat_monkey(
    verifier_system_prompt: str,
    system_prompt: typing.Union[str, None],
    user_prompt: str,
    num_monkeys: int = 3,
    schema: typing.Union[pydantic.BaseModel, dict, typing.Type[pydantic.BaseModel], None] = None,
    llm_args: typing.Dict[str, typing.Any] = {},
    verifier_llm_args: typing.Dict[str, typing.Any] = {},
):
    if DISABLE_ALL_SCHEMA:
        raise ValueError("DISABLE_ALL_SCHEMA and BIG_MODEL_CHAIN_OF_THOUGHT are both active.")
    monkey_res = await asyncio.gather(
        *[
            chat(
                system_prompt,
                user_prompt,
                model=DEFAULT_MODEL,
                schema=schema,
                llm_args={**{"temperature": 1}, **llm_args},
            )
            for _ in range(num_monkeys)
        ]
    )
    possible_answers = [f"Answer {string.ascii_uppercase[i]}" for i in range(num_monkeys)]

    class TheBestResponse(pydantic.BaseModel):
        the_best_response_is: typing.Literal[*possible_answers]  # type: ignore

    monkey_res_f = ("\n" * 5).join(
        ["-" * 20 + f"\n# {possible_answers[i]}\n" + str(monkey_res[i]) for i in range(num_monkeys)]
    )

    best_response = await chat(
        system_prompt=verifier_system_prompt,
        user_prompt=monkey_res_f,
        schema=TheBestResponse.model_json_schema(),
        llm_args={**{"temperature": 0}, **verifier_llm_args},
        model=BIG_MODEL,
    )
    assert isinstance(best_response, str)

    logger.info(
        f"From the {num_monkeys} monkeys with the results {monkey_res_f} the best response is {best_response}"
    )

    best_index = possible_answers.index(_str2dict(best_response)["the_best_response_is"])
    return monkey_res[best_index]


async def chat(
    system_prompt: typing.Union[str, None],
    user_prompt: str,
    schema: typing.Union[pydantic.BaseModel, dict, typing.Type[pydantic.BaseModel], None] = None,
    llm_args: typing.Union[typing.Dict[str, typing.Any], None] = None,
    model=None,
) -> typing.Union[str, pydantic.BaseModel]:
    if DISABLE_ALL_SCHEMA:
        if schema:
            logger.warning(f"Schema is disabled: {schema}")
            user_prompt += (
                "\n\nUse this schema:\n" + str(schema.model_json_schema())
                if isinstance(schema, pydantic.BaseModel)
                else str(schema)
            )
        schema = None
    assert not (model == BIG_MODEL and BIG_MODEL_CHAIN_OF_THOUGHT and schema), (
        f"cannot use schema with model {model} if chain of thought is enabled"
    )
    if model is None:
        model = DEFAULT_MODEL
    if llm_args is None:
        llm_args = {}
    if schema is None:
        rf = {}
    else:
        rf = {
            "response_format": {
                "type": "json_object" if USE_LLAMA_CPP_COMPATIBLE else "json_schema",
                "json_schema": {
                    "schema": schema.model_json_schema() if isinstance(schema, pydantic.BaseModel) else schema
                },
                "strict": True,
            }
        }

    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": user_prompt})
    logger.debug(
        f"Chat {model} {llm_args}\n\t{system_prompt.__repr__()}\n\t{user_prompt.__repr__()}\n\t{rf.__repr__()}"
    )

    try:
        completion = await CLIENT.beta.chat.completions.parse(
            model=model,
            messages=messages,  # type: ignore
            **rf,
            **llm_args,
        )
    except openai.LengthFinishReasonError as e:
        logger.warning(
            f"LengthFinishReasonError: {e} occurred, system_prompt: {repr(system_prompt)}, user_prompt: {repr(user_prompt)}"
        )
        raise e
    except Exception as e:
        logger.warning(
            f"Error: {e} occurred, system_prompt: {repr(system_prompt)}, user_prompt: {repr(user_prompt)}"
        )
        raise e

    if completion.usage and hasattr(completion.usage, "prompt_tokens"):
        if completion.usage.prompt_tokens in [2048, 4096]:
            logger.warning(f"Prompt tokens exhausted: {completion.usage.prompt_tokens}, {completion}")
            print(
                f"Prompt tokens exhausted, input sequence too long: {completion.usage.prompt_tokens}. Adjust Ollama settings"
            )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError(f"Empty response: {content}")
    if BIG_MODEL_CHAIN_OF_THOUGHT:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    if not isinstance(schema, pydantic.BaseModel):
        if USE_LLAMA_CPP_COMPATIBLE:
            return content
        return content
    else:
        try:
            validation = schema.model_validate_json(content)
        except pydantic.ValidationError as e:
            print(f"Validation Error: {e}, content: {content}, schema: {schema}")
            raise e
        return validation


async def list_models() -> typing.List[Model]:
    return (await CLIENT.models.list()).data


if __name__ == "__main__":
    for model in asyncio.run(list_models()):
        print(model.id)
