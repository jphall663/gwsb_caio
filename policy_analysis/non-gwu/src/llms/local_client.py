# Copyright (c) 2025 ph@hallresearch.ai
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Local (non-OpenAI) language-model helpers
=========================================

This module mirrors the conveniences provided by :mod:`hr_rag.llms.openai_client`
but targets self-hosted Hugging Face chat models. The exported helpers are:

``build_pipeline``
    Lazily import :mod:`torch` / :mod:`transformers`, load the tokenizer and
    model, and construct a text-generation pipeline with sensible defaults.
``danube_complete``
    Apply a Danube-style chat template to the provided prompt.

Example
-------

.. code-block:: python

    result = local_client_module.danube_complete('hello world')
    cut_idx = result[0]['generated_text'].find('<|answer|>')
    answer = result[0]['generated_text'][cut_idx + 10:]

"""

from __future__ import annotations

from typing import Any, Dict, List

from hr_rag import config

FALLBACK_MODEL = config.FALLBACK_MODEL
CLIENT_NAME = config.CLIENT_NAME

torch = None
transformers = None
AutoModelForCausalLM = None
AutoTokenizer = None

__all__ = ['build_pipeline', 'danube_complete']


def _require_dependencies() -> None:

    """Import torch/transformers on demand to keep pytest noise low."""

    global torch, transformers, AutoModelForCausalLM, AutoTokenizer

    if all([torch, transformers, AutoModelForCausalLM, AutoTokenizer]):
        return

    try:  # pragma: no cover - exercised indirectly
        import torch as torch_mod
        import transformers as transformers_mod
        from transformers import AutoModelForCausalLM as automodel_cls
        from transformers import AutoTokenizer as autotokenizer_cls
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'Local LLM support requires torch and transformers. '
            "Install the 'local' extras or ensure those packages are available."
        ) from exc

    torch = torch_mod
    transformers = transformers_mod
    AutoModelForCausalLM = automodel_cls
    AutoTokenizer = autotokenizer_cls


def build_pipeline(model: str = FALLBACK_MODEL):

    """
    Construct a Hugging Face text-generation pipeline.

    Parameters
    ----------
    model:
        Repository identifier passed directly to
        :func:`transformers.AutoModelForCausalLM.from_pretrained`. Defaults to
        :data:`hr_rag.config.FALLBACK_MODEL`.

    Returns
    -------
    ``transformers.Pipeline``
        A text-generation pipeline configured with BF16 precision and automatic
        device placement.
    """

    _require_dependencies()

    hf_tokenizer = AutoTokenizer.from_pretrained(
        model,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    return transformers.pipeline(
        'text-generation',
        tokenizer=hf_tokenizer,
        model=hf_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=False,
    )


def danube_complete(
    prompt: str,
    *,
    model: str = FALLBACK_MODEL,
    client_name: str = CLIENT_NAME,
    pipeline=None,
    max_new_tokens: int = 512,
) -> str:
    
    """
    Run a completion using a local Danube-style chat model.

    Parameters
    ----------
    prompt:
        The user prompt to feed into the chat template.
    model:
        Optional override for the HF repository name. Defaults to the configured
        fallback model.
    client_name:
        Name used in the system prompt when describing the analyst persona.
    pipeline:
        Optional pre-built Hugging Face pipeline. Supplying a pipeline speeds up
        repeated calls in environments where model loading is expensive (e.g.,
        tests or batch scoring). When omitted, :func:`build_pipeline` is used.
    max_new_tokens:
        Generation limit forwarded to the text-generation pipeline.

    Returns
    -------
    str
        The completed response. 

    """

    if pipeline is None:
        pipe = build_pipeline(model)
    else:
        pipe = pipeline

    messages = [
        {
            'role': 'system',
            'content': (
                f'You are a regulatory analyst at {client_name}. '
                f'Refer to yourself as we or {client_name}. Respond in a staid and measured tone.'
            ),
        },
        {'role': 'user', 'content': prompt},
    ]

    prompt_template = pipe.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs: List[Dict[str, Any]] = pipe(
        prompt_template,
        return_full_text=True,
        max_new_tokens=max_new_tokens,
    )

    return outputs[0].get('generated_text', '')
