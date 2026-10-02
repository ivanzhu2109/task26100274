"""In-process async workflow executor with first-response-wins joins.

Only stdlib. In-memory execution only.

Graph format: mapping of node_id to either a NodeSpec or a dict like
{"kind": "agent", "deps": ["a", "b"]}. Kinds:
  - "agent": runs handlers[node_id](inputs) where inputs maps predecessor
    ids to their outputs. Runs only after all predecessors succeed; fails
    immediately (without calling the handler) if any predecessor fails.
  - "first": no handler. The first accepted successful predecessor decides
    the output and winner_id; fails only when all predecessors fail.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping


class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


FINAL_STATUSES = frozenset(
    {Status.SUCCEEDED, Status.FAILED, Status.SKIPPED, Status.CANCELLED}
)


class DependencyFailedError(Exception):
    """Raised as a node's error when a predecessor failed."""


class AllPredecessorsFailedError(Exception):
    """Raised as a first-join's error when every predecessor failed."""


@dataclass(frozen=True)
class NodeSpec:
    kind: str  # "agent" or "first"
    deps: tuple[str, ...] = ()


@dataclass(frozen=True)
class NodeSnapshot:
    node_id: str
    status: Status
    output: Any = None
    error: BaseException | None = None
    winner_id: str | None = None


@dataclass(frozen=True)
class RunResult:
    output: Any
    states: dict[str, NodeSnapshot]


Handler = Callable[[dict[str, Any]], Awaitable[Any]]


def _normalize_spec(raw: Any) -> NodeSpec:
    if isinstance(raw, NodeSpec):
        return raw
    if isinstance(raw, Mapping):
        return NodeSpec(kind=raw["kind"], deps=tuple(raw.get("deps", ())))
    raise TypeError(f"unsupported node spec: {raw!r}")


class _NodeState:
    __slots__ = ("node_id", "spec", "status", "output", "error", "winner_id", "task")

    def __init__(self, node_id: str, spec: NodeSpec) -> None:
        self.node_id = node_id
        self.spec = spec
        self.status = Status.PENDING
        self.output: Any = None
        self.error: BaseException | None = None
        self.winner_id: str | None = None
        self.task: asyncio.Task | None = None

    def snapshot(self) -> NodeSnapshot:
        return NodeSnapshot(
            node_id=self.node_id,
            status=self.status,
            output=self.output,
            error=self.error,
            winner_id=self.winner_id,
        )


class Run:
    def __init__(
        self,
        graph: Mapping[str, Any],
        handlers: Mapping[str, Handler],
        output_id: str,
    ) -> None:
        self._states: dict[str, _NodeState] = {
            nid: _NodeState(nid, _normalize_spec(raw)) for nid, raw in graph.items()
        }
        if output_id not in self._states:
            raise KeyError(f"unknown output node: {output_id!r}")
        for nid, st in self._states.items():
            for dep in st.spec.deps:
                if dep not in self._states:
                    raise KeyError(f"node {nid!r} depends on unknown node {dep!r}")
            if st.spec.kind == "agent" and nid not in handlers:
                raise KeyError(f"missing handler for agent node {nid!r}")
        self._handlers = handlers
        self._output_id = output_id
        self._errors: list[BaseException] = []
        self._result_future: asyncio.Future[RunResult] = (
            asyncio.get_running_loop().create_future()
        )
        self._propagate()

    # ---- core engine -------------------------------------------------

    def _settle(
        self,
        nid: str,
        status: Status,
        output: Any = None,
        error: BaseException | None = None,
        winner_id: str | None = None,
    ) -> bool:
        st = self._states[nid]
        if st.status in FINAL_STATUSES:
            return False  # already decided; late results are discarded
        st.status = status
        st.output = output
        st.error = error
        st.winner_id = winner_id
        return True

    def _propagate(self) -> None:
        """Settle newly decidable nodes and start newly runnable agents."""
        changed = True
        while changed:
            changed = False
            for nid, st in self._states.items():
                if st.status is not Status.PENDING:
                    continue
                dep_states = [self._states[d] for d in st.spec.deps]
                if st.spec.kind == "agent":
                    failed = [d for d in dep_states if d.status is Status.FAILED]
                    if failed:
                        changed = self._settle(
                            nid,
                            Status.FAILED,
                            error=DependencyFailedError(
                                f"predecessor {failed[0].node_id!r} failed"
                            ),
                        ) or changed
                    elif all(d.status is Status.SUCCEEDED for d in dep_states):
                        self._start_agent(st)
                        changed = True
                else:  # first join
                    winners = [d for d in dep_states if d.status is Status.SUCCEEDED]
                    if winners:
                        winner = winners[0]
                        changed = self._settle(
                            nid,
                            Status.SUCCEEDED,
                            output=winner.output,
                            winner_id=winner.node_id,
                        ) or changed
                    elif dep_states and all(
                        d.status is Status.FAILED for d in dep_states
                    ):
                        changed = self._settle(
                            nid,
                            Status.FAILED,
                            error=AllPredecessorsFailedError(
                                f"all predecessors of {nid!r} failed"
                            ),
                        ) or changed
        self._recompute_needed()
        self._maybe_finish()

    def _start_agent(self, st: _NodeState) -> None:
        st.status = Status.RUNNING
        inputs = {d: self._states[d].output for d in st.spec.deps}
        st.task = asyncio.create_task(self._run_agent(st, inputs))

    async def _run_agent(self, st: _NodeState, inputs: dict[str, Any]) -> None:
        handler = self._handlers[st.node_id]
        try:
            output = await handler(inputs)
        except asyncio.CancelledError:
            raise  # executor-requested cancellation; nothing to collect
        except BaseException as exc:
            self._errors.append(exc)
            self._settle(st.node_id, Status.FAILED, error=exc)
        else:
            self._settle(st.node_id, Status.SUCCEEDED, output=output)
        self._propagate()

    def _recompute_needed(self) -> None:
        """Skip or logically cancel undecided nodes no longer needed.

        A node is still needed iff there is a path from it to the output
        node along which every node (including itself) is undecided.
        """
        undecided = {
            nid
            for nid, st in self._states.items()
            if st.status in (Status.PENDING, Status.RUNNING)
        }
        needed: set[str] = set()
        stack = [self._output_id] if self._output_id in undecided else []
        while stack:
            cur = stack.pop()
            if cur in needed:
                continue
            needed.add(cur)
            stack.extend(
                d for d in self._states[cur].spec.deps
                if d in undecided and d not in needed
            )
        for nid in undecided - needed:
            st = self._states[nid]
            if st.status is Status.PENDING:
                self._settle(nid, Status.SKIPPED)
            else:  # RUNNING -> logical cancel, request real cancellation
                self._settle(nid, Status.CANCELLED)
                if st.task is not None:
                    st.task.cancel()

    def _maybe_finish(self) -> None:
        if self._result_future.done():
            return
        out = self._states[self._output_id]
        if out.status not in FINAL_STATUSES:
            return
        snapshot = {nid: st.snapshot() for nid, st in self._states.items()}
        self._result_future.set_result(RunResult(output=out.output, states=snapshot))

    # ---- public API ----------------------------------------------------

    async def result(self) -> RunResult:
        """Wait for the output node to settle; return output + stable snapshot."""
        return await asyncio.shield(self._result_future)

    async def wait_idle(self) -> list[BaseException]:
        """Wait until every started callback has truly exited.

        Returns the exceptions raised by callbacks during this run
        (including late exceptions from logically cancelled nodes).
        """
        seen: set[asyncio.Task] = set()
        while True:
            tasks = [
                st.task
                for st in self._states.values()
                if st.task is not None and st.task not in seen
            ]
            if not tasks:
                return list(self._errors)
            seen.update(tasks)
            await asyncio.gather(*tasks, return_exceptions=True)


def start(
    graph: Mapping[str, Any],
    handlers: Mapping[str, Handler],
    output_id: str,
) -> Run:
    """Start a run and return its Run handle immediately.

    Must be called inside a running event loop.
    """
    return Run(graph, handlers, output_id)
