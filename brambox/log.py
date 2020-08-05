#
#   Copyright EAVISE
#
import logging
import os


__all__ = ['set_log_level', 'logger']


# Deprecation level
def deprecated(self, message, *args, **kws):
    if not hasattr(self, 'deprecated_msgs'):
        self.deprecated_msgs = []

    if self.isEnabledFor(35) and message not in self.deprecated_msgs:
        self.deprecated_msgs.append(message)
        self._log(35, message, args, **kws)


logging.addLevelName(35, "DEPRECATED")
logging.Logger.deprecated = deprecated

# Console Handler
ch = logging.StreamHandler()
if 'BB_LOGLVL' in os.environ:
    ch.setLevel(os.environ['BB_LOGLVL'])
else:
    ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(levelname)s [%(name)s]  %(message)s'))
set_log_level = ch.setLevel

# Logger
logger = logging.getLogger('brambox')
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)
