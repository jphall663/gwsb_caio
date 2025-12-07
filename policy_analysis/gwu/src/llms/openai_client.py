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
hr_rag.llms.openai_client module
================================

High-level helpers for interacting with the OpenAI API, including:

- a shared client instance configured from :mod:`config`,
- default language model parameters for chat completions,
- convenience functions for embeddings and completions,
- a simple retry mechanism with exponential backoff.

This module assumes that the following symbols are defined in a separate
:mod:`config` module:

- ``API_KEY``
- ``OPENAI_API_TIMEOUT``
- ``CLIENT_NAME``
- ``EMBEDDING_MODEL``
- ``OPENAI_API_RETRIES``
- ``REMOTE_MODEL``
- ``SEED``

Quickstart
----------

Basic usage for embeddings and completions:

.. code-block:: python

    from hr_rag.llms.openai_client import gpt_embed, gpt_complete

    text = 'OpenAI provides powerful language models.'
    embedding = gpt_embed(text)
    print(f'Embedding length: {len(embedding)}')

    prompt = 'Summarize the importance of regulatory compliance in finance.'
    response = gpt_complete(prompt)
    print(response.choices[0].message.content)


Configuration
-------------

The module reads configuration from :mod:`config` and uses it to construct a
shared OpenAI client and default completion parameters.

"""

from copy import deepcopy
from functools import wraps

import config


API_KEY = config.API_KEY
OPENAI_API_TIMEOUT = config.OPENAI_API_TIMEOUT
CLIENT_NAME = config.CLIENT_NAME
EMBEDDING_MODEL = config.EMBEDDING_MODEL
OPENAI_API_RETRIES = config.OPENAI_API_RETRIES
REMOTE_MODEL = config.REMOTE_MODEL
SEED = config.SEED
import openai
import random
import time

from logging_utils import get_logger
logger = get_logger(__name__)

""" 
Language model parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autodata:: hr_rag.llms.openai_client.lm_params

Dictionary of default parameters used for chat completions, for example:

- ``model``: remote chat model name from :data:`config.REMOTE_MODEL`
- ``messages``: initial system and user message structure, including the
  regulatory analyst persona and client name (:data:`config.CLIENT_NAME`)
- ``frequency_penalty``: numeric penalty applied to frequent tokens
- ``seed``: deterministic seed from :data:`config.SEED`
- ``temperature``: sampling temperature
"""

lm_params = {
   'model': REMOTE_MODEL,
   'messages': [
         {'role': 'system',
         'content': f'''You are a regulatory analyst assistant. 
         Respond in a staid and measured tone. 
         If you refer to yourself, do so as "we" or {CLIENT_NAME}.'''},
         {'role': 'user',
          'content': None}
   ],
   'frequency_penalty': -0.3,
   'seed': SEED,
   'temperature': 0.2
}

"""

Client
~~~~~~

.. autodata:: hr_rag.llms.openai_client.client
   :annotation: = openai.OpenAI(...)

The global OpenAI client instance configured with:

- ``api_key`` from :data:`config.API_KEY`
- ``timeout`` from :data:`config.OPENAI_API_TIMEOUT`

"""

client = openai.OpenAI(api_key=API_KEY, timeout=OPENAI_API_TIMEOUT)

def retry_with_exponential_backoff(
    func=None,
    *,
    logger=None,
    initial_delay:float=1,
    exponential_base:float=2,
    jitter:bool=True,
    max_retries:int=OPENAI_API_RETRIES,
    errors:tuple=(openai.RateLimitError,),
    verbose:bool=False,
):
      
   """
   Wrap a callable with retry logic using exponential backoff.

   This decorator may be invoked with or without arguments. When used bare
   (``@retry_with_exponential_backoff``) the default logger and retry settings
   are applied. When called with keyword arguments (for example,
   ``@retry_with_exponential_backoff(max_retries=3)``) the returned decorator
   configures the wrapped function accordingly.

   Parameters
   ----------
   func
      The callable to wrap and retry. Typically a function that
      makes an OpenAI API request. May be ``None`` when the decorator
      is configured with keyword arguments.
   logger
      Logger instance used to emit informational and error messages
      about retries and failures. Defaults to :func:`hr_rag.logging_utils.get_logger`
      using the wrapped function's module.
   initial_delay : float, optional
      Initial delay in seconds before the first retry. Defaults to ``1``.
   exponential_base : float, optional
      Base for the exponential increase of the delay. Each retry
      multiplies the current delay by this factor (and jitter, if enabled).
      Defaults to ``2``.
   jitter : bool, optional
      If ``True``, multiply the delay by a random factor to introduce
      jitter. Defaults to ``True``.
   max_retries : int, optional
      Maximum number of retry attempts before giving up and raising an
      exception. Defaults to :data:`config.OPENAI_API_RETRIES`.
   errors : tuple, optional
      Tuple of exception types that should trigger a retry. Any other
      exception type is treated as non-retriable but still subject to the
      same max-retry check. Defaults to ``(openai.RateLimitError,)``.
   verbose : bool, optional
      If ``True``, emit informational log messages before each attempt and
      retry, including the current retry count and next delay. Defaults to
      ``False``.

   Returns
   -------
   callable
      A wrapped version of ``func`` that will be retried according to the
      configured backoff policy.

   Raises
   ------
   Exception
      If the maximum number of retries is exceeded, the original exception is
      re-raised so stack information is preserved.
   """

   def decorator(f):
      
      log = logger or get_logger(f.__module__)
      retries_allowed = max(max_retries, 0)

      @wraps(f)
      def wrapper(*args, **kwargs):
           
         num_retries = 0
         delay = initial_delay
         tic = time.time()

         while True:

            try:

               if verbose:
                  log.info('Polling OpenAI API ...')
                  log.info(f'Current retries: {num_retries}/{retries_allowed}; Next delay: {delay:.2f} s. ...')

               return f(*args, **kwargs)

            except errors as err:

               last_exc = err

            except Exception as err:

               last_exc = err
               toc = time.time() - tic
               log.error(f'An error has occurred after {toc:.2f} s. when using the OpenAI API: "{err}".')

            num_retries += 1

            if num_retries > retries_allowed:
               raise last_exc

            delay *= exponential_base * (1 + jitter * random.random())
            time.sleep(delay)

      return wrapper

   if func is None:
      return decorator

   return decorator(func)

@retry_with_exponential_backoff
def gpt_embed(text:str) -> list:

   """
   Create an embedding vector for the given text using the configured model.

   This function calls the OpenAI embeddings API via the shared client
   and returns the embedding vector for the input text. It is wrapped by
   :func:`retry_with_exponential_backoff`, so transient errors (such as
   rate limits) will be retried according to the configured policy.

   Parameters
   ----------
   text : str
      Input text to be converted into an embedding vector.

   Returns
   -------
   list
      A list of floats representing the embedding vector returned by the
      embeddings API for the given ``text``.
   """

   return client.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding

@retry_with_exponential_backoff
def gpt_complete(prompt, lm_params_=None):
   
   """
   Generate a chat completion response for the given prompt.

   This function copies either the module-level :data:`lm_params` template
   or the provided ``lm_params_`` so the call can mutate the user-entry
   message safely. After the prompt is inserted, it calls the OpenAI chat
   completions API via
   the shared client. It is wrapped by
   :func:`retry_with_exponential_backoff`, so transient errors will be
   retried according to the configured backoff policy.

   Parameters
   ----------
   prompt
      The user prompt to send to the language model. This value will be
      inserted into the copied template's user message (index ``1``) before
      the request is made.
   lm_params_ : dict, optional
      Dictionary of language-model parameters passed directly to
      ``client.chat.completions.create``. By default, this is the module-
      level :data:`lm_params`, which includes model name, initial messages,
      temperature, seed, and other settings. Custom dictionaries are deep-
      copied before mutation so the caller's data remains unchanged.

   Returns
   -------
   Any
      The response object returned by :func:`client.chat.completions.create`.
      Typically this is an OpenAI chat completion response containing one
      or more choices.
   """

   params_template = lm_params if lm_params_ is None else lm_params_
   params = deepcopy(params_template)
   messages = params.get('messages')

   if not messages or len(messages) < 2:
      raise ValueError('lm_params must define at least two messages (system + user).')

   params['messages'] = [dict(message) for message in messages]
   params['messages'][1]['content'] = prompt

   return client.chat.completions.create(**params)
