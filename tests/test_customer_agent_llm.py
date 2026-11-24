import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from pydantic import BaseModel

from Agent.CustomerAgent.custom.customer_agent import CustomerAgent
from Agent.CustomerAgent.custom import customer_agent as customer_agent_module
from Agent.CustomerAgent.custom.agent_config import AgentConfig
from Agent.CustomerAgent.custom.tool_decorator import TOOL_REGISTRY, agent_tool
from Agent.CustomerAgent.custom.llm_client import LLMResponse
from Agent.CustomerAgent.custom.session_manager import SessionManager
from utils.llm_provider import LLMProfile, LLMProvider
from utils.llm_transport import NormalizedFunction, NormalizedToolCall


class _LookupParams(BaseModel):
    value: str


class CustomerAgentRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialization_passes_one_validated_profile_snapshot_to_client(self):
        profile_config = AgentConfig(
            provider="deepseek",
            model_name="deepseek-chat",
            api_key="key",
            api_base="",
        )

        class FakeClient:
            instances = []

            def __init__(self, **kwargs):
                self.profile = kwargs["profile"]
                self.tools = []
                self.initialized = False
                FakeClient.instances.append(self)

            async def initialize(self):
                self.initialized = True

            async def close(self):
                pass

        fake_session = mock.Mock()
        with mock.patch.object(
            customer_agent_module.AgentConfig,
            "load_from_config",
            return_value=profile_config,
        ), mock.patch.object(customer_agent_module, "LLMClient", FakeClient), mock.patch.object(
            customer_agent_module, "SessionManager", return_value=fake_session
        ):
            agent = CustomerAgent()
            self.assertTrue(await agent.initialize_async())
            self.assertTrue(FakeClient.instances[0].initialized)
            self.assertEqual(agent._active_profile.provider, LLMProvider.DEEPSEEK)
            self.assertIs(agent._active_profile, FakeClient.instances[0].profile)
            await agent.close()

    async def test_multi_round_tool_loop_preserves_transcript_and_profile_snapshot(self):
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        tool_name = "__test_customer_agent_lookup__"

        @agent_tool(
            name=tool_name,
            description="test lookup",
            param_model=_LookupParams,
        )
        def lookup(params: _LookupParams) -> str:
            return f"lookup:{params.value}"

        responses = [
            AIMessage(
                content="checking",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": {"value": "x"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="final answer"),
        ]

        class ScriptedModel(BaseChatModel):
            calls: list = []

            @property
            def _llm_type(self) -> str:
                return "scripted"

            def bind_tools(self, tools, *, tool_choice=None, **kwargs):
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                raise NotImplementedError

            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                ScriptedModel.calls.append(list(messages))
                return ChatResult(
                    generations=[ChatGeneration(message=responses.pop(0))]
                )

        ScriptedModel.calls = []

        with TemporaryDirectory() as directory:
            session = SessionManager(str(Path(directory) / "session.db"))
            try:
                agent = object.__new__(CustomerAgent)
                agent._config = SimpleNamespace(max_loops=3, temperature=0.3)
                agent._session_manager = session
                profile = LLMProfile(
                    provider=LLMProvider.DEEPSEEK,
                    model_name="deepseek-chat",
                    api_key="key",
                )
                agent._active_profile = profile
                messages = [{"role": "user", "content": "please lookup"}]

                with mock.patch.object(
                    agent, "_build_chat_model", return_value=ScriptedModel()
                ):
                    result = await agent._run_agent_loop(
                        messages,
                        {},
                        session_id="session-1",
                    )

                self.assertEqual(result, "final answer")
                self.assertEqual(len(ScriptedModel.calls), 2)
                history = session.get_history("session-1")
                self.assertEqual(
                    json.loads(history[0]["content"])["tool_calls"][0]["id"],
                    "call-1",
                )
                self.assertEqual(history[1]["tool_call_id"], "call-1")
                self.assertEqual(history[1]["content"], "lookup:x")
            finally:
                session.dispose()
                TOOL_REGISTRY.pop(tool_name, None)

    async def test_summary_operation_explicitly_disables_tools(self):
        calls = []

        class FakeClient:
            async def chat(self, messages, **kwargs):
                calls.append(kwargs)
                return LLMResponse("summary", [], object(), {})

        class FakeSession:
            async def compress_history(self, session_id, callback):
                await callback([{"role": "user", "content": "history"}])

        agent = object.__new__(CustomerAgent)
        agent._llm_client = FakeClient()
        agent._session_manager = FakeSession()
        await agent._compress_with_llm("session-1")
        self.assertEqual(calls[0]["tool_choice"], "none")
        self.assertFalse(calls[0]["use_tools"])


if __name__ == "__main__":
    unittest.main()
