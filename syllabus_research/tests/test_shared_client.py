# Copyright (c) 2025-2026 Patrick Hall, jphall@gwu.edu
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from shared.llms import openai_client


class Recorder:
    def __init__(self):
        self.chat_kwargs = None
        self.response_kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.chat_create))
        self.responses = SimpleNamespace(create=self.response_create)

    def chat_create(self, **kwargs):
        self.chat_kwargs = kwargs
        return SimpleNamespace(choices=[])

    def response_create(self, **kwargs):
        self.response_kwargs = kwargs
        return SimpleNamespace(output_text="ok")


def test_chat_completion_contract_remains_compatible(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(openai_client, "client", recorder)
    params = {
        "model": "legacy-model",
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": None}],
        "temperature": 0.2,
    }
    result = openai_client.gpt_complete("hello", params)
    assert result.choices == []
    assert recorder.chat_kwargs["messages"][1]["content"] == "hello"
    assert params["messages"][1]["content"] is None


def test_responses_api_reasoning_and_tool_configuration_are_additive(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(openai_client, "client", recorder)
    result = openai_client.gpt_response(
        "find courses",
        model="reasoning-model",
        reasoning_effort="medium",
        tools=[{"type": "web_search"}],
        tool_choice="required",
        max_output_tokens=1234,
    )
    assert result.output_text == "ok"
    assert recorder.response_kwargs["reasoning"] == {"effort": "medium"}
    assert recorder.response_kwargs["tools"] == [{"type": "web_search"}]
    assert recorder.response_kwargs["tool_choice"] == "required"
    assert recorder.response_kwargs["max_output_tokens"] == 1234
