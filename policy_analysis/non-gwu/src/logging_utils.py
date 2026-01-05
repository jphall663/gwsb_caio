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
logging_utils.py
================

A simple logging configuration utility providing colorized console output
and safe re-import protection.

This module offers a convenience function :func:`configure_logging` to set up
logging either to the console (with colored levels) or to a file (plain text).
It also provides a helper :func:`get_logger` to safely obtain a configured
logger instance.

Example
-------

.. code-block:: python

    from logging_utils import configure_logging, get_logger
    import logging

    # Configure logging to console with colored output
    configure_logging(level=logging.DEBUG)

    logger = get_logger(__name__)
    logger.debug('Debug details')
    logger.info('All good')
    logger.warning('This is a warning')
    logger.error('Something went wrong!')
    logger.critical('Fatal problem!')

    # Configure logging to file instead
    # configure_logging(level=logging.INFO, to_file='app.log')

"""

import logging
from typing import Optional
import sys

_CONFIGURED = False  # prevent duplicate handlers

class ColorFormatter(logging.Formatter):

    """
    A log formatter that colorizes the log level name for console output.

    This formatter applies ANSI color codes to the log level (``DEBUG``,
    ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``) to make console logs
    easier to read. When used with a file handler, a regular
    :class:`logging.Formatter` should be used instead.

    Attributes
    ----------
    COLORS : dict[int, str]
        Mapping from logging levels to ANSI color escape sequences.
    RESET : str
        ANSI reset code to clear formatting after the level name.
    """

    COLORS = {
        logging.DEBUG: '\033[36m',   # cyan
        logging.INFO: '\033[32m',    # green
        logging.WARNING: '\033[33m', # yellow
        logging.ERROR: '\033[31m',   # red
        logging.CRITICAL: '\033[41m' # red background
    }

    RESET = '\033[0m'

    def format(self, record:logging.LogRecord) -> str:
        
         """
         Format the log record, coloring the level name.

         Parameters
         ----------
         record : logging.LogRecord
            The log record to format.

         Returns
         -------
         str
            The formatted log message with colorized level name.
         """

         level_color = self.COLORS.get(record.levelno, '')
         levelname_colored = f'{level_color}{record.levelname}{self.RESET}'
         record.levelname = levelname_colored
         return super().format(record)

def configure_logging(*, # all following args must be passed by keywords
                      level:int=logging.INFO, to_file: Optional[str]=None, 
                      fmt:str='%(asctime)s %(levelname)s [%(name)s]: %(message)s',
                      datefmt:str='%Y-%m-%d %H:%M:%S') -> None:
    
    """
    Configure the root logger once per application run.

    By default, logs are sent to the console with colorized level names.
    If ``to_file`` is provided, logs are written to that file instead
    (without colors).

    Parameters
    ----------
    level : int, optional
        Logging level to use (e.g., :data:`logging.DEBUG`,
        :data:`logging.INFO`). Defaults to ``logging.INFO``.
    to_file : str, optional
        Path to a log file. If ``None`` (default), logs are written to
        ``stderr``.
    fmt : str, optional
        Log message format string. Defaults to
        ``'%(asctime)s %(levelname)s [%(name)s]: %(message)s'``.
    datefmt : str, optional
        Date format string for timestamps. Defaults to ``'%Y-%m-%d %H:%M:%S'``.

    Notes
    -----
    - Repeated calls to this function have no effect after the first call.
      This prevents duplicate handlers being added on module re-import.
    """

    global _CONFIGURED

    if _CONFIGURED:
        return  # don't reconfigure

    logger = logging.getLogger()
    logger.setLevel(level)

    if to_file:
        handler = logging.FileHandler(to_file, encoding='utf-8')
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    else:
        handler = logging.StreamHandler(sys.stderr)
        formatter = ColorFormatter(fmt=fmt, datefmt=datefmt)

    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    _CONFIGURED = True

def get_logger(name:Optional[str]=None) -> logging.Logger:

    """
    Retrieve a configured logger.

    Ensures that logging is configured before returning the requested logger.
    If :func:`configure_logging` has not yet been called, this function will
    invoke it with default settings.

    Parameters
    ----------
    name : str, optional
        Logger name. If ``None``, the root logger is returned.

    Returns
    -------
    logging.Logger
        The requested logger instance.
    """

    global _CONFIGURED
    
    if not _CONFIGURED:
        configure_logging()  # default configuration
    
    return logging.getLogger(name)
