from enum import Enum

class ReadFileError(Enum):
    METADATA_NOT_FOUND = "No metadata matches the given ID."
    FILE_NOT_FOUND = "File not found."

class ErrorType(Enum):
    NO_FACE_DETECED = "No face detected"
    FACE_NOT_FOUND = "Face not found"
    NOT_MOVING_FACE = "Not a moving face"
    INTERNAL_SERVER_ERROR = "Internal server error"
    USER_EXISTED = "the given user information has already exist in database"
    FACE_EXISTED = "the given image contains a face that has already exist in database"
    IMAGE_NOT_VALIDATE = "This image has not been validated"
    IMAGE_HAS_BEEN_USED = "This image has been used"
    USER_FACE_NOT_MATCH = "The given user information and face aren't matched"

class STATUS(Enum):
    SUCCEED = "Succeed"
    FAILED = "Failed"