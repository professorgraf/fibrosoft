

class MLProcessingException(Exception):
    def __init__(self, message="Unknown ML processing error.") -> None:
        super().__init__(message)


class MLFileTypeException(Exception):
    def __init__(self, message="Wrong Filetype") -> None:
        super().__init__(message)
