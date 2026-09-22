import logging

# Library default: emit nothing until an application asks for logs. Anything
# that wants output calls portfolio_agent.logging_setup.setup_logging().
logging.getLogger(__name__).addHandler(logging.NullHandler())
