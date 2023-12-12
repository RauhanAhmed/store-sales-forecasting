import sys

def error_message_detail(error):
    _, _, exc_tb = sys.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    error_message = "Error encountered in line number [{}], filename [{}] saying : [{}]".format(lineno, filename, error)
    return error_message


class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_message_detail(error = error_message)

    def __str__(self) -> str:
        return self.error_message