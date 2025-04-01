recognize_face = {
    200: {
        "description": "Face recognized successfully",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Succeed",
                        "message": "Found a face matches the given data",
                        "code": 200,
                        "data": {"face_id": 123, "similarity": 0.95}
                    }
                }
            }
        }
    },
    404: {
        "description": "Face not found or file not found",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "Face not found",
                        "code": 404
                    }
                }
            }
        }
    },
    400: {
        "description": "Bad request due to invalid image",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "No face detected",
                        "code": 400
                    }
                }
            }
        }
    }
}

add_embedding = {
    200: {
        "description": "Face embedding added successfully",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Succeed",
                        "message": "Face embedding added successfully!",
                        "code": 200
                    }
                }
            }
        }
    },
    409: {
        "description": "Image already used by another user",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "This image has been used",
                        "code": 409
                    }
                }
            }
        }
    },
    404: {
        "description": "Metadata not found",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "No metadata matches the given ID.",
                        "code": 404
                    }
                }
            }
        }
    },
    400: {
        "description": "Bad request due to invalid image or unmatched user-face",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "No face detected",
                        "code": 400
                    }
                }
            }
        }
    }
}

image_validate = {
    200: {
        "description": "Face validated successfully",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Succeed",
                        "message": "Face validated successfully!",
                        "code": 200
                    }
                }
            }
        }
    },
    409: {
        "description": "Face already exists",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "the given image contains a face that has already exist in database",
                        "code": 409
                    }
                }
            }
        }
    },
    500: {
        "description": "Internal server error",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "Internal server error",
                        "code": 500
                    }
                }
            }
        }
    },
    404: {
        "description": "Metadata or file not found",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "No metadata matches the given ID.",
                        "code": 404
                    }
                }
            }
        }
    }
}

upload_image = {
    200: {
        "description": "Image uploaded successfully",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Succeed",
                        "message": "Upload image successfully!",
                        "code": 200,
                        "data": {
                            "image_id": "12345",  # Ví dụ về kết quả dữ liệu trả về từ save_image
                            "filename": "image.jpg",
                            "url": "https://example.com/images/image.jpg"
                        }
                    }
                }
            }
        }
    },
    400: {
        "description": "Invalid file type",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "Only image files (JPG, JPEG, PNG, GIF, BMP, WEBP) are allowed.",
                        "code": 400
                    }
                }
            }
        }
    },
    500: {
        "description": "Internal server error",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "Internal server error!",
                        "code": 500
                    }
                }
            }
        }
    }
}

delete_user = {
    200: {
        "description": "User deleted successfully",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Succeed",
                        "message": "Deleted user successfully.",
                        "code": 200
                    }
                }
            }
        }
    },
    500: {
        "description": "Internal server error",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "Internal Server error.",
                        "code": 500
                    }
                }
            }
        }
    }
}

add_user = {
    200: {
        "description": "User deleted successfully",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Succeed",
                        "message": "Added user successfully.",
                        "code": 200
                    }
                }
            }
        }
    },
    500: {
        "description": "Internal server error",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "status": "Failed",
                        "message": "Internal Server error.",
                        "code": 500
                    }
                }
            }
        }
    }
}