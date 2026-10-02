import asyncio
import unittest

from workflow import Kind, NodeSpec, Run, State, WorkflowError, start


def agent(*preds):
    return NodeSpec(Kind.AGENT, tuple(preds))


def first(*preds):
    return NodeSpec(Kind.FIRST, tuple(preds))


class TestBasicExecution(unittest.IsolatedAsyncioTestCase):
    async def test_chain_passes_outputs_by_pred_id(self):
        graph = {"a": agent(), "b": agent("a"), "c": agent("b")}
        seen = {}

        async def ha(inputs):
            seen["a"] = inputs
            return 1

        async def hb(inputs):
            seen["b"] = inputs
            return inputs["a"] + 1

        async def hc(inputs):
            seen["c"] = inputs
            return inputs["b"] + 1

        run = start(graph, {"a": ha, "b": hb, "c": hc}, "c")
        value, snap = await run.result()
        self.assertEqual(value, 3)
        self.assertEqual(seen, {"a": {}, "b": {"a": 1}, "c": {"b": 2}})
        self.assertTrue(all(s.state is State.SUCCEEDED for s in snap.values()))
        self.assertEqual(await run.wait_idle(), [])

    async def test_diamond_shared_dependency_runs_once(self):
        graph = {"a": agent(), "b": agent("a"), "c": agent("a"), "d": agent("b", "c")}
        calls = []

        async def ha(inputs):
            calls.append("a")
            return "A"

        async def hb(inputs):
            return inputs["a"] + "b"

        async def hc(inputs):
            return inputs["a"] + "c"

        async def hd(inputs):
            return (inputs["b"], inputs["c"])

        run = start(graph, {"a": ha, "b": hb, "c": hc, "d": hd}, "d")
        value, _ = await run.result()
        self.assertEqual(value, ("Ab", "Ac"))
        self.assertEqual(calls, ["a"])

    async def test_agent_failure_skips_descendant_callback(self):
        graph = {"a": agent(), "b": agent("a")}
        called = []
        boom = RuntimeError("boom")

        async def ha(inputs):
            raise boom

        async def hb(inputs):
            called.append("b")

        run = start(graph, {"a": ha, "b": hb}, "b")
        with self.assertRaises(WorkflowError):
            await run.result()
        self.assertEqual(called, [])
        self.assertEqual(await run.wait_idle(), [boom])

    async def test_first_join_fails_only_when_all_preds_fail(self):
        graph = {"a": agent(), "b": agent(), "j": first("a", "b")}

        async def fail(inputs):
            raise ValueError("nope")

        run = start(graph, {"a": fail, "b": fail}, "j")
        with self.assertRaises(WorkflowError):
            await run.result()
        errors = await run.wait_idle()
        self.assertEqual(len(errors), 2)

    async def test_first_join_winner_is_stable(self):
        graph = {"a": agent(), "b": agent(), "j": first("a", "b")}

        async def ha(inputs):
            return "a"

        async def hb(inputs):
            await asyncio.sleep(0.05)
            return "b"

        run = start(graph, {"a": ha, "b": hb}, "j")
        value, snap = await run.result()
        self.assertEqual(value, "a")
        self.assertEqual(snap["j"].winner_id, "a")
        await run.wait_idle()
        _, snap2 = await run.result()
        self.assertEqual(snap2["j"].winner_id, "a")


class TestLoserStillNeeded(unittest.IsolatedAsyncioTestCase):
    async def test_loser_on_other_path_continues_and_runs_once(self):
        graph = {
            "a": agent(),
            "b": agent(),
            "j": first("a", "b"),
            "c": agent("b"),
            "t": agent("j", "c"),
        }
        b_go = asyncio.Event()
        b_started = asyncio.Event()
        b_calls = []

        async def ha(inputs):
            return "a-out"

        async def hb(inputs):
            b_calls.append(1)
            b_started.set()
            await b_go.wait()
            return "b-out"

        async def hc(inputs):
            return ("c", inputs["b"])

        async def ht(inputs):
            return {"join": inputs["j"], "via_c": inputs["c"]}

        run = start(graph, {"a": ha, "b": hb, "c": hc, "t": ht}, "t")
        await asyncio.wait_for(b_started.wait(), timeout=1)
        await asyncio.sleep(0)  # let the join settle on a's win
        b_go.set()
        value, snap = await run.result()
        self.assertEqual(value, {"join": "a-out", "via_c": ("c", "b-out")})
        self.assertEqual(b_calls, [1])
        self.assertEqual(snap["j"].winner_id, "a")
        self.assertIs(snap["b"].state, State.SUCCEEDED)
        self.assertEqual(await run.wait_idle(), [])


class TestLoserCancelled(unittest.IsolatedAsyncioTestCase):
    def make_graph(self):
        return {"a": agent(), "b": agent(), "j": first("a", "b")}

    def make_handlers(self, b_started, release, cancelled, exited, late_error):
        async def ha(inputs):
            await b_started.wait()  # ensure b's callback is running before a wins
            return "a-out"

        async def hb(inputs):
            b_started.set()
            try:
                await asyncio.Event().wait()  # never completes
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                exited.set()
                if late_error is not None:
                    raise late_error
                return "late-b"
            return "b-out"

        return {"a": ha, "b": hb}

    async def test_late_success_is_discarded(self):
        b_started, release = asyncio.Event(), asyncio.Event()
        cancelled, exited = asyncio.Event(), asyncio.Event()
        handlers = self.make_handlers(b_started, release, cancelled, exited, late_error=None)
        run = start(self.make_graph(), handlers, "j")
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        value, snap = await asyncio.wait_for(run.result(), timeout=1)
        self.assertEqual(value, "a-out")
        self.assertEqual(snap["j"].winner_id, "a")
        self.assertIs(snap["b"].state, State.CANCELLED)
        self.assertFalse(exited.is_set())  # result before loser exits
        release.set()
        self.assertEqual(await asyncio.wait_for(run.wait_idle(), timeout=1), [])
        self.assertTrue(exited.is_set())
        _, snap2 = await run.result()
        self.assertEqual(snap2, snap)  # late success changed nothing

    async def test_late_exception_is_discarded_but_collected(self):
        b_started, release = asyncio.Event(), asyncio.Event()
        cancelled, exited = asyncio.Event(), asyncio.Event()
        boom = RuntimeError("late boom")
        handlers = self.make_handlers(b_started, release, cancelled, exited, late_error=boom)
        run = start(self.make_graph(), handlers, "j")
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        value, snap = await asyncio.wait_for(run.result(), timeout=1)
        self.assertEqual(value, "a-out")
        self.assertIs(snap["b"].state, State.CANCELLED)
        release.set()
        errors = await asyncio.wait_for(run.wait_idle(), timeout=1)
        self.assertEqual(errors, [boom])
        _, snap2 = await run.result()
        self.assertEqual(snap2, snap)  # late exception changed nothing


if __name__ == "__main__":
    unittest.main()
