import sys 
from src.FlightPrice_predictor.logger import logging

def error_message_detail(error,error_detail: sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in Python script{file_name} at lineno {exc_tb.tb_lineno} with error{str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details)
        #super().__init__(self.error_message)

    def __str__(self):
        return self.error_message



# import sys
# import logging

# def error_message_detail(error, error_detail: sys):
#     _, _, exc_tb = error_detail.exc_info()
#     file_name = exc_tb.tb_frame.f_code.co_filename
#     error_message = (
#         f"Error occurred in Python script [{file_name}] "
#         f"at line [{exc_tb.tb_lineno}] "
#         f"with error: {str(error)}"
#     )
#     return error_message

# class CustomException(Exception):
#     def __init__(self, error, error_details: sys):
#         # Build full error message
#         self.error_message = error_message_detail(error, error_details)
#         # Pass message to base Exception
#         super().__init__(self.error_message)

#     def __str__(self):
#         return self.error_message
