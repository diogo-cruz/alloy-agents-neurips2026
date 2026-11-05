"""Inspect-compatible alloy solver that alternates between models per turn.

This is the core contribution of the alloy safety framework. It provides solvers:

- `alloy_generate`: Single-turn generation that selects a model per call. Use this as a
  drop-in replacement for `generate()` in solver chains.
- `alloy_agent`: Multi-turn agentic solver that runs a tool loop, alternating models
  each turn. Mirrors task_generate() behavior.
- `alloy_basic_agent`: Multi-turn agentic solver with submit/retry logic, alternating
  models each turn. Mirrors basic_agent() behavior.

All solvers bypass Inspect's default `generate()` (which always uses the task's model)
and instead call `get_model(model_name).generate()` directly to select different models
per turn.
"""

from __future__ import annotations

import base64
from typing import Awaitable, Callable, cast

from pydantic import Field

from inspect_ai._util._async import is_callable_coroutine
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.model._call_tools import execute_tools
from inspect_ai.scorer._metric import Score, ValueToFloat, value_to_float
from inspect_ai.scorer._score import score
from inspect_ai.solver import Generate, Solver, TaskState, chain, solver, system_message, use_tools
from inspect_ai.solver._basic_agent import (
    DEFAULT_CONTINUE_MESSAGE,
    DEFAULT_INCORRECT_MESSAGE,
    DEFAULT_SUBMIT_DESCRIPTION,
    DEFAULT_SUBMIT_NAME,
)
from inspect_ai.tool import Tool, ToolResult, tool
from inspect_ai.tool._tool_with import tool_with
from inspect_ai.util import StoreModel, span, store_as

from alloy_safety.strategies import FixedSequence, RoundRobin, Roulette, SelectionStrategy

# Google's official dummy value to bypass thought_signature validation.
# Gemini 3+ requires thought_signature on functionCall parts. When tool calls
# come from other providers (GPT, Claude), they lack this field.
# Inspect's google.py converts ContentReasoning(redacted=True) to thought_signature.
_DUMMY_THOUGHT_SIG = base64.urlsafe_b64encode(
    b"skip_thought_signature_validator"
).decode()


class AlloyState(StoreModel):
    """Tracks alloy solver state per sample. Auto-logged to transcript via store."""

    turn_count: int = Field(default=0)
    model_sequence: list[str] = Field(default_factory=list)


def _make_strategy(
    strategy: str,
    weights: list[float] | None = None,
    sequence: list[int] | None = None,
    seed: int | None = None,
) -> SelectionStrategy:
    """Create a selection strategy from string name."""
    if strategy == "round_robin":
        return RoundRobin()
    elif strategy == "roulette":
        return Roulette(weights=weights, seed=seed)
    elif strategy == "fixed_sequence":
        if sequence is None:
            raise ValueError("fixed_sequence strategy requires a sequence parameter")
        return FixedSequence(sequence=sequence)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _resolve_models(model_names: list[str], config: GenerateConfig | None) -> list[Model]:
    """Resolve model name strings to Model objects."""
    return [get_model(name, config=config or GenerateConfig()) for name in model_names]


def _log_turn(turn: int, model_name: str, strategy_name: str) -> None:
    """Log alloy turn metadata to the Inspect transcript."""
    transcript().info(
        data={
            "event": "alloy_turn",
            "turn": turn,
            "model": model_name,
            "strategy": strategy_name,
        },
        source="alloy_solver",
    )


def _is_homogeneous(models: list[str]) -> bool:
    """Check if all models in the alloy are identical."""
    return len(set(models)) == 1


def _inject_thought_signatures(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Inject dummy thought signatures for Gemini compatibility.

    Gemini 3+ models require thought_signature on functionCall parts in the
    message history. When tool calls come from other providers (GPT, Claude),
    they lack this metadata. This injects Google's official dummy signature
    ("skip_thought_signature_validator") via a redacted ContentReasoning block,
    which Inspect's Google provider converts to thought_signature on the Part.

    Must be called AFTER stripping redacted reasoning.
    """
    result: list[ChatMessage] = []
    for msg in messages:
        if isinstance(msg, ChatMessageAssistant) and msg.tool_calls:
            dummy = ContentReasoning(
                reasoning=_DUMMY_THOUGHT_SIG,
                redacted=True,
            )
            if isinstance(msg.content, str):
                new_content = ([ContentText(text=msg.content)] if msg.content else []) + [dummy]
            else:
                new_content = list(msg.content) + [dummy]
            msg = msg.model_copy(update={"content": new_content})
        result.append(msg)
    return result


def _prepare_messages(
    messages: list[ChatMessage],
    homogeneous: bool,
    target_model: str = "",
) -> list[ChatMessage]:
    """Prepare messages for the next model in the alloy.

    - Homogeneous alloy (A+A): return messages unchanged — model sees its own
      prior reasoning, matching solo behavior exactly.
    - Heterogeneous alloy (A+B): strip only encrypted (redacted) reasoning.
      Open reasoning is kept, allowing models to benefit from each other's
      chain-of-thought when it's readable. Then inject dummy thought signatures
      if the target is a Gemini model (required by Gemini 3+).
    """
    if homogeneous:
        return messages  # match solo: let model.generate() handle reasoning_history

    result: list[ChatMessage] = []
    for msg in messages:
        if isinstance(msg, ChatMessageAssistant) and not isinstance(msg.content, str):
            new_content = [
                c for c in msg.content
                if not (isinstance(c, ContentReasoning) and c.redacted)
            ]
            if new_content != msg.content:
                msg = msg.model_copy(update={"content": new_content or ""})
        result.append(msg)

    # Gemini 3+ requires thought_signature on functionCall parts
    if target_model.startswith("google/"):
        result = _inject_thought_signatures(result)

    return result


# Global counter for distributing single-turn samples across models.
# This is shared across all samples in a run so that round-robin
# actually alternates which model handles which sample.
_global_sample_counter: int = 0


def _next_global_turn() -> int:
    """Get and increment the global sample counter."""
    global _global_sample_counter
    turn = _global_sample_counter
    _global_sample_counter += 1
    return turn


def reset_global_counter() -> None:
    """Reset the global sample counter (call between experiment runs)."""
    global _global_sample_counter
    _global_sample_counter = 0


@solver
def alloy_generate(
    models: list[str],
    strategy: str = "round_robin",
    weights: list[float] | None = None,
    sequence: list[int] | None = None,
    seed: int | None = None,
    config: GenerateConfig | None = None,
) -> Solver:
    """Single-turn alloy solver. Selects a model and generates one response.

    Use this as a drop-in replacement for `generate()` in solver chains.
    Uses a global counter so that round-robin distributes samples across models
    (e.g., sample 0 -> model A, sample 1 -> model B, sample 2 -> model A, ...).

    For multi-turn conversations within a single sample, use `alloy_agent` instead.

    Args:
        models: List of model names, e.g. ["openai/gpt-4o", "google/gemini-2.5-flash"].
        strategy: "round_robin", "roulette", or "fixed_sequence".
        weights: Weights for roulette strategy.
        sequence: Model index sequence for fixed_sequence strategy.
        seed: Random seed for roulette strategy.
        config: GenerateConfig applied to all generations.
    """
    sel_strategy = _make_strategy(strategy, weights, sequence, seed)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        resolved = _resolve_models(models, config)
        alloy_state = store_as(AlloyState)

        # Use global counter for cross-sample distribution
        turn = _next_global_turn()
        idx, model = sel_strategy.select(turn, resolved)
        model_name = models[idx]

        alloy_state.turn_count = 1
        alloy_state.model_sequence = [model_name]
        _log_turn(turn + 1, model_name, sel_strategy.name())

        output: ModelOutput = await model.generate(
            input=state.messages,
            tools=state.tools,
            config=config or GenerateConfig(),
        )

        state.messages.append(output.message)
        state.output = output
        return state

    return solve


@solver
def alloy_agent(
    models: list[str],
    strategy: str = "round_robin",
    max_messages: int = 100,
    weights: list[float] | None = None,
    sequence: list[int] | None = None,
    seed: int | None = None,
    config: GenerateConfig | None = None,
) -> Solver:
    """Multi-turn agentic alloy solver with tool loop.

    Mirrors task_generate() behavior: each cycle generates one response and
    executes any tool calls, then swaps to the next model. The loop ends when
    the model produces no tool calls or max_messages is reached.

    For A+A homogeneous alloys, this is identical to solo task_generate()
    (messages passed unchanged, matching solo reasoning_history behavior).

    Args:
        models: List of model names.
        strategy: "round_robin", "roulette", or "fixed_sequence".
        max_messages: Safety valve on total messages (not a behavioral constraint).
        weights: Weights for roulette strategy.
        sequence: Model index sequence for fixed_sequence strategy.
        seed: Random seed for roulette strategy.
        config: GenerateConfig applied to all generations.
    """
    sel_strategy = _make_strategy(strategy, weights, sequence, seed)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        resolved = _resolve_models(models, config)
        homogeneous = _is_homogeneous(models)
        alloy_state = store_as(AlloyState)
        gen_config = config or GenerateConfig()
        turn = 0

        while not state.completed and len(state.messages) < max_messages:
            idx, model = sel_strategy.select(turn, resolved)
            model_name = models[idx]

            alloy_state.turn_count = turn + 1
            alloy_state.model_sequence = [*alloy_state.model_sequence, model_name]
            _log_turn(turn + 1, model_name, sel_strategy.name())

            async with span(f"alloy_turn_{turn + 1}"):
                output = await model.generate(
                    input=_prepare_messages(state.messages, homogeneous, model_name),
                    tools=state.tools,
                    tool_choice=state.tool_choice,
                    config=gen_config,
                )
                state.messages.append(output.message)
                state.output = output

                if state.completed:
                    break

                if output.message.tool_calls:
                    tool_result = await execute_tools(
                        state.messages, state.tools, gen_config.max_tool_output
                    )
                    state.messages.extend(tool_result.messages)
                    if tool_result.output is not None:
                        state.output = tool_result.output
                    if state.completed:
                        break
                    turn += 1
                else:
                    break

        return state

    return solve


@solver
def alloy_basic_agent(
    models: list[str],
    *,
    init: Solver | list[Solver] | None = None,
    tools: list[Tool] | Solver | None = None,
    strategy: str = "round_robin",
    max_attempts: int = 1,
    message_limit: int | None = None,
    max_tool_output: int | None = None,
    score_value: ValueToFloat | None = None,
    incorrect_message: str
    | Callable[
        [TaskState, list[Score]], str | Awaitable[str]
    ] = DEFAULT_INCORRECT_MESSAGE,
    continue_message: str = DEFAULT_CONTINUE_MESSAGE,
    submit_name: str = DEFAULT_SUBMIT_NAME,
    submit_description: str = DEFAULT_SUBMIT_DESCRIPTION,
    weights: list[float] | None = None,
    sequence: list[int] | None = None,
    seed: int | None = None,
    config: GenerateConfig | None = None,
) -> Solver:
    """Multi-turn alloy agent with submit/retry logic.

    Mirrors basic_agent() but alternates models each turn using the alloy
    selection strategy. Use for benchmarks that need submit/retry behavior
    (e.g., ImpossibleBench).

    For A+A homogeneous alloys, this is identical to solo basic_agent()
    (messages passed unchanged, matching solo reasoning_history behavior).

    Args:
        models: List of model names.
        init: Agent initialisation (defaults to system_message with basic ReAct prompt).
        tools: Tools available for the agent.
        strategy: "round_robin", "roulette", or "fixed_sequence".
        max_attempts: Maximum number of submissions to accept before terminating.
        message_limit: Limit on messages in sample before terminating agent.
        max_tool_output: Maximum output length (in bytes).
        score_value: Function used to extract float from scores.
        incorrect_message: User message reply for an incorrect submission.
        continue_message: User message to urge the model to continue.
        submit_name: Name for tool used to make submissions.
        submit_description: Description of submit tool.
        weights: Weights for roulette strategy.
        sequence: Model index sequence for fixed_sequence strategy.
        seed: Random seed for roulette strategy.
        config: GenerateConfig applied to all generations.
    """
    sel_strategy = _make_strategy(strategy, weights, sequence, seed)

    # resolve init
    if init is None:
        from inspect_ai.solver._basic_agent import DEFAULT_SYSTEM_MESSAGE

        init = system_message(DEFAULT_SYSTEM_MESSAGE, submit=submit_name)
    init = init if isinstance(init, list) else [init]

    # resolve tools
    if tools is None:
        tools = []
    tools_solver = tools if isinstance(tools, Solver) else use_tools(tools, append=True)

    # resolve score_value function
    score_value_fn = score_value or value_to_float()

    # submission tool
    @tool
    def submit() -> Tool:
        async def execute(answer: str) -> ToolResult:
            """Submit an answer for evaluation.

            Args:
              answer (str): Submitted answer
            """
            return answer

        return execute

    # solver that adds submission tool
    @solver
    def submit_tool() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.tools.append(tool_with(submit(), submit_name, submit_description))
            return state

        return solve

    # helper to extract a submitted answer
    def submission(tool_results: list[ChatMessage]) -> str | None:
        return next(
            (
                result.text
                for result in tool_results
                if isinstance(result, ChatMessageTool)
                and result.function == submit_name
            ),
            None,
        )

    # main agent loop with alloy model selection
    @solver
    def alloy_loop() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            resolved = _resolve_models(models, config)
            homogeneous = _is_homogeneous(models)
            alloy_state = store_as(AlloyState)
            gen_config = config or GenerateConfig()

            # resolve message_limit
            state.message_limit = message_limit or state.message_limit
            if state.message_limit is None:
                state.message_limit = 50

            turn = 0
            attempts = 0

            while not state.completed:
                idx, model = sel_strategy.select(turn, resolved)
                model_name = models[idx]

                alloy_state.turn_count = turn + 1
                alloy_state.model_sequence = [*alloy_state.model_sequence, model_name]
                _log_turn(turn + 1, model_name, sel_strategy.name())

                async with span(f"alloy_turn_{turn + 1}"):
                    state.output = await model.generate(
                        input=_prepare_messages(state.messages, homogeneous, model_name),
                        tools=state.tools,
                        config=gen_config,
                    )
                    state.messages.append(state.output.message)

                    if state.output.stop_reason == "model_length":
                        transcript().info(
                            "Agent terminated: model context window exceeded"
                        )
                        break

                    if state.output.message.tool_calls:
                        tool_results, _ = await execute_tools(
                            [state.output.message],
                            state.tools,
                            max_output=max_tool_output,
                        )
                        state.messages.extend(tool_results)
                        turn += 1

                        answer = submission(tool_results)
                        if answer:
                            state.output.completion = answer
                            attempts += 1
                            if attempts >= max_attempts:
                                break

                            answer_scores = await score(state)
                            if score_value_fn(answer_scores[0].value) == 1.0:
                                break

                            if is_callable_coroutine(incorrect_message):
                                response_message: str = await incorrect_message(
                                    state, answer_scores
                                )  # type: ignore[misc,operator]
                            elif callable(incorrect_message):
                                response_message = cast(
                                    str, incorrect_message(state, answer_scores)
                                )
                            else:
                                response_message = incorrect_message

                            state.messages.append(
                                ChatMessageUser(content=response_message)
                            )
                    else:
                        state.messages.append(
                            ChatMessageUser(content=continue_message)
                        )
                        turn += 1

            return state

        return solve

    return chain(init + [tools_solver, submit_tool(), alloy_loop()])
