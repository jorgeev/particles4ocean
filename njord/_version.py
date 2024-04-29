import logging
from importlib.metadata import PackageNotFoundError, distribution

_logger = logging.getLogger(__name__)

try:
    version = distribution("njord").version
except PackageNotFoundError:
	version = "0000"
	_logger.info(F"Cannot determine package version using {version}")
