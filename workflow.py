"""In-process async workflow executor with first-response-wins joins."""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


class Kind(enum.Enum):
    AGENT = "agent"
    FIRST = "first"


class State(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


_DECIDED = frozenset({State.SUCCEEDED, State.FAILED, State.SKIPPED, State.CANCELLED})
_DEAD_PREDS = frozenset({State.FAILED, State.SKIPPED, State.CANCELLED})


@dataclass(frozen=True)
class NodeSpec:
    kind: Kind
    preds: Tuple[str, ...] = ()


@dataclass(frozen=True)
class NodeSnapshot:
    state: State
    output: Any = None
    winner_id: Optional[str] = None


class WorkflowError(Exception):
    pass


Handler = Callable[[Dict[str, Any]], Awaitable[Any]]


class Run:
    def __init__(self, graph: Dict[str, NodeSpec], handlers: Dict[str, Handler], output_id: str) -> None:
        if output_id not in graph:
            raise ValueError(f"unknown output node: {output_id!r}")
        for nid, spec in graph.items():
            if spec.kind is Kind.AGENT and nid not in handlers:
                raise ValueError(f"missing handler for agent node: {nid!r}")
            for pred in spec.preds:
                if pred not in graph:
                    raise ValueError(f"node {nid!r} has unknown predecessor {pred!r}")
        self._graph = dict(graph)
        self._handlers = handlers
        self._output_id = output_id
        self._states: Dict[str, State] = {nid: State.PENDING for nid in graph}
        self._outputs: Dict[str, Any] = {}
        self._winners: Dict[str, str] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._all_tasks: List[asyncio.Task] = []
        self._exceptions: List[BaseException] = []
        self._output_event = asyncio.Event()
        self._successors: Dict[str, List[str]] = {nid: [] for nid in graph}
        for nid, spec in graph.items():
            for pred in spec.preds:
                self._successors[pred].append(nid)
        self._refresh_needed()
        for nid in self._graph:
            if self._states[nid] is State.PENDING:
                self._maybe_trigger(nid)

    async def result(self) -> Tuple[Any, Dict[str, NodeSnapshot]]:
        await self._output_event.wait()
        state = self._states[self._output_id]
        if state is not State.SUCCEEDED:
            raise WorkflowError(f"output node {self._output_id!r} settled as {state.value}")
        return self._outputs[self._output_id], self._snapshot()

    async def wait_idle(self) -> List[BaseException]:
        while True:
            tasks = [t for t in self._all_tasks if not t.done()]
            if not tasks:
                return list(self._exceptions)
            await asyncio.gather(*tasks, return_exceptions=True)

    def _snapshot(self) -> Dict[str, NodeSnapshot]:
        return {
            nid: NodeSnapshot(
                state=self._states[nid],
                output=self._outputs.get(nid),
                winner_id=self._winners.get(nid),
            )
            for nid in self._graph
        }

    def _settle(self, nid: str, state: State, output: Any = None, winner: Optional[str] = None) -> None:
        if self._states[nid] in _DECIDED:
            return
        self._states[nid] = state
        if state is State.SUCCEEDED:
            self._outputs[nid] = output
            if winner is not None:
                self._winners[nid] = winner
        self._refresh_needed()
        for succ in self._successors[nid]:
            if self._states[succ] is State.PENDING:
                self._maybe_trigger(succ)
        if nid == self._output_id:
            self._output_event.set()

    def _maybe_trigger(self, nid: str) -> None:
        spec = self._graph[nid]
        pred_states = [self._states[p] for p in spec.preds]
        if spec.kind is Kind.AGENT:
            if any(s in _DEAD_PREDS for s in pred_states):
                self._settle(nid, State.FAILED)
            elif all(s is State.SUCCEEDED for s in pred_states):
                self._start_agent(nid)
        else:
            alive_winners = [
                p for p, s in zip(spec.preds, pred_states) if s is State.SUCCEEDED
            ]
            if alive_winners:
                winner = alive_winners[0]
                self._settle(nid, State.SUCCEEDED, output=self._outputs[winner], winner=winner)
            elif all(s in _DEAD_PREDS for s in pred_states):
                self._settle(nid, State.FAILED)

    def _start_agent(self, nid: str) -> None:
        self._states[nid] = State.RUNNING
        task = asyncio.get_running_loop().create_task(self._run_agent(nid))
        self._tasks[nid] = task
        self._all_tasks.append(task)

    async def _run_agent(self, nid: str) -> None:
        inputs = {p: self._outputs[p] for p in self._graph[nid].preds}
        try:
            value = await self._handlers[nid](inputs)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._exceptions.append(exc)
            self._settle(nid, State.FAILED)
        else:
            self._settle(nid, State.SUCCEEDED, output=value)

    def _refresh_needed(self) -> None:
        undecided = {nid for nid, s in self._states.items() if s not in _DECIDED}
        needed: set = set()
        if self._states[self._output_id] not in _DECIDED:
            stack = [self._output_id]
            while stack:
                nid = stack.pop()
                if nid in needed or nid not in undecided:
                    continue
                needed.add(nid)
                stack.extend(self._graph[nid].preds)
        for nid in undecided - needed:
            if self._states[nid] is State.PENDING:
                self._states[nid] = State.SKIPPED
            elif self._states[nid] is State.RUNNING:
                self._states[nid] = State.CANCELLED
                self._tasks[nid].cancel()


def start(graph: Dict[str, NodeSpec], handlers: Dict[str, Handler], output_id: str) -> Run:
    return Run(graph, handlers, output_id)
