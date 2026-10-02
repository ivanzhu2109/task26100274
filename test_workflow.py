import asyncio
import unittest

from workflow import start, Status


async def pump(rounds: int = 20) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


class SharedDependencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_loser_still_needed_runs_once(self):
        a_release = asyncio.Event()
        b_release = asyncio.Event()
        b_started = asyncio.Event()
        b_cancelled = asyncio.Event()
        b_calls = 0

        async def a_handler(inputs):
            assert inputs == {}
            await a_release.wait()
            return "a"

        async def b_handler(inputs):
            nonlocal b_calls
            b_calls += 1
            b_started.set()
            try:
                await b_release.wait()
            except asyncio.CancelledError:
                b_cancelled.set()
                raise
            return "b"

        async def c_handler(inputs):
            return ("c", inputs["B"])

        async def out_handler(inputs):
            return dict(inputs)

        graph = {
            "A": {"kind": "agent", "deps": []},
            "B": {"kind": "agent", "deps": []},
            "F": {"kind": "first", "deps": ["A", "B"]},
            "C": {"kind": "agent", "deps": ["B"]},
            "OUT": {"kind": "agent", "deps": ["F", "C"]},
        }
        run = start(
            graph,
            {"A": a_handler, "B": b_handler, "C": c_handler, "OUT": out_handler},
            "OUT",
        )

        await asyncio.wait_for(b_started.wait(), timeout=1)
        a_release.set()
        await pump()

        # A won the join, but B is still needed via C and must not be cancelled.
        self.assertEqual(b_calls, 1)
        self.assertFalse(b_cancelled.is_set())

        b_release.set()
        result = await asyncio.wait_for(run.result(), timeout=1)

        self.assertEqual(b_calls, 1)  # each node runs at most once per run
        self.assertEqual(result.output, {"F": "a", "C": ("c", "b")})
        self.assertEqual(result.states["F"].winner_id, "A")
        self.assertEqual(result.states["F"].status, Status.SUCCEEDED)
        self.assertEqual(result.states["B"].status, Status.SUCCEEDED)
        self.assertEqual(result.states["B"].output, "b")
        self.assertEqual(result.states["OUT"].status, Status.SUCCEEDED)

        errors = await asyncio.wait_for(run.wait_idle(), timeout=1)
        self.assertEqual(errors, [])


class CancelledLoserTest(unittest.IsolatedAsyncioTestCase):
    async def test_loser_catches_cancel_late_success(self):
        b_started = asyncio.Event()
        caught = asyncio.Event()
        release = asyncio.Event()

        async def a_handler(inputs):
            await b_started.wait()  # win only after B has actually started
            return "a"

        async def b_handler(inputs):
            b_started.set()
            try:
                await asyncio.Event().wait()  # never set: blocks until cancelled
            except asyncio.CancelledError:
                caught.set()
            await release.wait()  # loser keeps running after catching cancel
            return "b-late"

        graph = {
            "A": {"kind": "agent", "deps": []},
            "B": {"kind": "agent", "deps": []},
            "F": {"kind": "first", "deps": ["A", "B"]},
        }
        run = start(graph, {"A": a_handler, "B": b_handler}, "F")

        await asyncio.wait_for(b_started.wait(), timeout=1)
        # Output settles while the loser is still running (post-cancel).
        result = await asyncio.wait_for(run.result(), timeout=1)
        self.assertEqual(result.output, "a")
        self.assertEqual(result.states["F"].winner_id, "A")
        self.assertEqual(result.states["B"].status, Status.CANCELLED)

        await asyncio.wait_for(caught.wait(), timeout=1)
        # Late success must not change the settled snapshot.
        release.set()
        errors = await asyncio.wait_for(run.wait_idle(), timeout=1)
        self.assertEqual(errors, [])
        self.assertEqual(result.states["B"].status, Status.CANCELLED)
        self.assertIsNone(result.states["B"].output)
        self.assertEqual(result.states["F"].output, "a")

    async def test_loser_catches_cancel_late_failure(self):
        b_started = asyncio.Event()
        caught = asyncio.Event()
        release = asyncio.Event()
        boom = RuntimeError("late boom")

        async def a_handler(inputs):
            await b_started.wait()  # win only after B has actually started
            return "a"

        async def b_handler(inputs):
            b_started.set()
            try:
                await asyncio.Event().wait()  # never set: blocks until cancelled
            except asyncio.CancelledError:
                caught.set()
            await release.wait()  # loser keeps running after catching cancel
            raise boom

        graph = {
            "A": {"kind": "agent", "deps": []},
            "B": {"kind": "agent", "deps": []},
            "F": {"kind": "first", "deps": ["A", "B"]},
        }
        run = start(graph, {"A": a_handler, "B": b_handler}, "F")

        await asyncio.wait_for(b_started.wait(), timeout=1)
        # Output settles while the loser is still running (post-cancel).
        result = await asyncio.wait_for(run.result(), timeout=1)
        self.assertEqual(result.output, "a")
        self.assertEqual(result.states["F"].winner_id, "A")
        self.assertEqual(result.states["B"].status, Status.CANCELLED)

        await asyncio.wait_for(caught.wait(), timeout=1)
        # Late failure must not change the settled snapshot.
        release.set()
        errors = await asyncio.wait_for(run.wait_idle(), timeout=1)
        self.assertEqual(errors, [boom])
        self.assertEqual(result.states["B"].status, Status.CANCELLED)
        self.assertIsNone(result.states["B"].error)
        self.assertEqual(result.states["F"].output, "a")


if __name__ == "__main__":
    unittest.main()
