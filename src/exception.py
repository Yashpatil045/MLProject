## for the exception handling purposes
import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    This function returns the error message with details.
    :param error: The exception object.
    :param error_detail: The sys module to get the exception type and value.
    :return: A string containing the error message with details.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: [{file_name}] at line number: [{exc_tb.tb_lineno}] with error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    """
    Custom exception class that inherits from the built-in Exception class.
    It overrides the constructor to provide a detailed error message.
    """
    def __init__(self, error, error_detail: sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message
    