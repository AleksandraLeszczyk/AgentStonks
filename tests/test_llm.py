import json
from types import SimpleNamespace

import anthropic
import pytest
from pydantic import BaseModel, Field

from agent_stonks import llm
from agent_stonks import observability as obs

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Latest quote.",
            "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}},
        },
    }
]


class _BadRequest(Exception):
    """Stands in for openai.BadRequestError: same status_code / param attributes."""

    def __init__(self, param):
        super().__init__(f"400 on {param}")
        self.status_code = 400
        self.param = param


class _FakeChatCompletions:
    def __init__(self, error=None):
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content="chat", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeResponses:
    def __init__(self, output=None, output_text=""):
        self.calls: list[dict] = []
        self.output = output or []
        self.output_text = output_text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output=self.output, output_text=self.output_text)


def _openai_client(chat_error=None, responses=None):
    raw = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeChatCompletions(chat_error)),
        responses=responses or _FakeResponses(),
    )
    return raw, llm._OpenAIToolChatClient(raw)


@pytest.fixture(autouse=True)
def _fresh_routing(monkeypatch):
    monkeypatch.setattr(llm, "_RESPONSES_API_MODELS", set())


class TestOpenAIToolRouting:
    def test_tool_call_on_chat_completions_gets_reasoning_none(self):
        raw, client = _openai_client()
        client.chat.completions.create(model="gpt-6-luna", messages=[], tools=_TOOLS, tool_choice="auto")
        assert raw.chat.completions.calls[0]["reasoning_effort"] == "none"
        assert raw.responses.calls == []

    def test_tool_less_call_is_passed_through_untouched(self):
        raw, client = _openai_client()
        client.chat.completions.create(model="gpt-6-astra", messages=[])
        assert "reasoning_effort" not in raw.chat.completions.calls[0]

    def test_model_refusing_reasoning_none_is_routed_to_responses_and_remembered(self):
        raw, client = _openai_client(chat_error=_BadRequest("reasoning_effort"))
        kwargs = dict(model="gpt-6-astra", messages=[{"role": "user", "content": "hi"}], tools=_TOOLS, tool_choice="auto")

        client.chat.completions.create(**kwargs)
        client.chat.completions.create(**kwargs)

        assert len(raw.chat.completions.calls) == 1  # the second call skips chat-completions
        assert len(raw.responses.calls) == 2
        assert "reasoning_effort" not in raw.responses.calls[0]
        assert llm._RESPONSES_API_MODELS == {"gpt-6-astra"}

    def test_other_bad_requests_are_raised(self):
        raw, client = _openai_client(chat_error=_BadRequest("messages"))
        with pytest.raises(_BadRequest):
            client.chat.completions.create(model="gpt-6-luna", messages=[], tools=_TOOLS, tool_choice="auto")
        assert raw.responses.calls == []


class TestResponsesAdapter:
    def test_chat_transcript_is_translated_to_responses_items(self):
        responses = _FakeResponses()
        adapter = llm._OpenAIResponsesCompletions(SimpleNamespace(responses=responses))
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_quote", "arguments": '{"symbol": "AAPL"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"price": 1}'},
        ]

        adapter.create(model="gpt-6-astra", messages=messages, tools=_TOOLS, tool_choice="auto")

        request = responses.calls[0]
        assert request["instructions"] == "sys"
        assert request["store"] is False
        assert request["input"] == [
            {"role": "user", "content": "go"},
            {"type": "function_call", "call_id": "call_1", "name": "get_quote", "arguments": '{"symbol": "AAPL"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": '{"price": 1}'},
        ]
        assert request["tools"] == [
            {
                "type": "function",
                "name": "get_quote",
                "description": "Latest quote.",
                "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}},
                "strict": False,
            }
        ]

    def test_function_calls_come_back_in_chat_completions_shape(self):
        output = [
            SimpleNamespace(type="reasoning"),
            SimpleNamespace(type="function_call", call_id="call_9", name="get_quote", arguments='{"symbol": "MU"}'),
        ]
        adapter = llm._OpenAIResponsesCompletions(SimpleNamespace(responses=_FakeResponses(output=output)))

        message = adapter.create(model="gpt-6-astra", messages=[], tools=_TOOLS).choices[0].message

        assert message.content is None
        assert [(tc.id, tc.function.name, json.loads(tc.function.arguments)) for tc in message.tool_calls] == [
            ("call_9", "get_quote", {"symbol": "MU"})
        ]


class _Verdict(BaseModel):
    score: int = Field(ge=0, le=10)
    note: str


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _anthropic(monkeypatch, *, text, stop_reason="end_turn"):
    messages = _FakeMessages(
        SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason, usage=None)
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: SimpleNamespace(messages=messages))
    monkeypatch.setattr(obs, "_client", lambda: None)
    return messages


class TestAnthropicStructured:
    def test_uses_structured_outputs_not_forced_tool_choice(self, monkeypatch):
        messages = _anthropic(monkeypatch, text='{"score": 7, "note": "ok"}')

        result = llm.parse_structured("anthropic", "k", "claude-sonnet-5-5", "sys", "user", _Verdict)

        assert result == _Verdict(score=7, note="ok")
        request = messages.calls[0]
        assert "tool_choice" not in request and "tools" not in request
        assert request["output_config"]["format"]["type"] == "json_schema"
        assert request["output_config"]["format"]["schema"]["additionalProperties"] is False

    @pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
    def test_refused_or_truncated_answer_is_no_result(self, monkeypatch, stop_reason):
        _anthropic(monkeypatch, text='{"score": 7', stop_reason=stop_reason)
        assert llm.parse_structured("anthropic", "k", "claude-sonnet-5-5", "sys", "user", _Verdict) is None

    def test_constraints_the_api_cannot_enforce_are_still_validated(self, monkeypatch):
        _anthropic(monkeypatch, text='{"score": 11, "note": "too high"}')
        with pytest.raises(ValueError):
            llm.parse_structured("anthropic", "k", "claude-sonnet-5-5", "sys", "user", _Verdict)
