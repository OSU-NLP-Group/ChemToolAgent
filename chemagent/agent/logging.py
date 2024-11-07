import logging

print_logger = logging.getLogger('chemagent_print')
print_logger.setLevel(logging.INFO)
print_handler = logging.StreamHandler()
print_formatter = logging.Formatter("%(message)s")
print_handler.setFormatter(print_formatter)
print_logger.addHandler(print_handler)
print_logger.propagate = False
