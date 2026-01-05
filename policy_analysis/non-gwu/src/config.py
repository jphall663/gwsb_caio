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
config.py
===================

Configuration constants and default values used throughout the project.

This module contains only global constants and configuration dictionaries.
It does not define any functions or classes.
"""

import os

# backend
SEED: int               = 12345 
EMBEDDING_P: int        = 1536
CHUNK_LENGTH: int       = 64
REMOTE_MODEL: str       = 'gpt-4o'
EMBEDDING_MODEL: str    = 'text-embedding-ada-002'
FALLBACK_MODEL: str     = 'h2oai/h2o-danube3.1-4b-chat'

# UI
CLIENT_NAME             = 'The George Washington School of Business'

### Open AI API
API_KEY: str            = os.environ['OPENAI_API_KEY']
OPENAI_API_TIMEOUT: int = 600
OPENAI_API_RETRIES: int = 0

