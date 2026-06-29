"""
logging.py — pychartai logging configuration.

Provides a centralized logger for the pychartai package,
replacing any PandasAI logging with pychartai-specific handlers.
"""

import logging
import sys
from typing import Optional

# Module-level logger
_logger: Optional[logging.Logger] = None


def get_logger(name: str = 'pychartai', level: int = logging.INFO) -> logging.Logger:
	"""
	Get or create the pychartai logger.

	Args:
		name: Logger name (default: 'pychartai')
		level: Logging level (default: logging.INFO)

	Returns:
		Configured logger instance
	"""
	global _logger

	if _logger is not None:
		return _logger

	_logger = logging.getLogger(name)
	_logger.setLevel(level)

	# Prevent duplicate handlers if logger is accessed multiple times
	if _logger.handlers:
		return _logger

	# Console handler with format
	handler = logging.StreamHandler(sys.stdout)
	handler.setLevel(level)

	formatter = logging.Formatter(
		'[pychartai] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
	)
	handler.setFormatter(formatter)

	_logger.addHandler(handler)

	# Suppress PandasAI's logger (don't propagate)
	_logger.propagate = False

	return _logger


def set_log_level(level: int) -> None:
	"""Set the logging level for pychartai logger."""
	logger = get_logger()
	logger.setLevel(level)
	for handler in logger.handlers:
		handler.setLevel(level)


def configure_logger(
	name: str = 'pychartai',
	level: int = logging.INFO,
	file_path: Optional[str] = '.pychartai.log',
) -> logging.Logger:
	"""
	Configure pychartai logger with optional file output.

	Args:
		name: Logger name
		level: Logging level
		file_path: Optional file path for file logging (default: .pychartai.log)

	Returns:
		Configured logger
	"""
	logger = get_logger(name, level)

	if file_path:
		file_handler = logging.FileHandler(file_path)
		file_handler.setLevel(level)
		formatter = logging.Formatter(
			'[pychartai] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S',
		)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	return logger


# Suppress PandasAI logging by default
def suppress_pandasai_logging() -> None:
	"""Suppress verbose PandasAI logging and remove any file handlers it creates."""
	import os
	pandasai_loggers = ['pandasai', 'pandasai.llm', 'pandas_ai', 'pandas_ai.llm']
	for lib in pandasai_loggers:
		lg = logging.getLogger(lib)
		lg.setLevel(logging.WARNING)
		# Remove any file handlers pandasai auto-creates (e.g. pandasai.log)
		for handler in list(lg.handlers):
			if isinstance(handler, logging.FileHandler):
				lg.removeHandler(handler)
				handler.close()
	# Also suppress other verbose libraries
	for lib in ['urllib3', 'requests']:
		logging.getLogger(lib).setLevel(logging.WARNING)
	# Remove pandasai.log file if it was already created
	for log_file in ['pandasai.log', 'pandas_ai.log']:
		try:
			if os.path.exists(log_file):
				os.remove(log_file)
		except OSError:
			pass


# NOTE: suppress_pandasai_logging() is NOT called automatically on import.
# It is called lazily from pychartai_core/__init__.py after pandasai availability check.


if __name__ == '__main__':
	logger = get_logger()
	logger.info('pychartai logging initialized')
	logger.debug('Debug message example')
	logger.warning('Warning message example')
