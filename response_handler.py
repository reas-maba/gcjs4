import time
import traceback
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Any, Optional, Dict
import logging

# 全局日志
logger = logging.getLogger("DogRecResponse")

# 标准返回结构模型
class StandardResponse(BaseModel):
    code: int
    msg: str
    data: Optional[Any] = None
    timestamp: int

# 状态码常量定义
CODE_SUCCESS = 200
CODE_PARAM_ERROR = 400
CODE_NOT_FOUND = 404
CODE_SERVER_ERROR = 500
CODE_UPLOAD_ERROR = 410
CODE_MODEL_ERROR = 501

class ResponseHandler:
    """全局统一返回封装、异常捕获、日志打印工具类"""
    @staticmethod
    def success(data: Any = None, msg: str = "请求成功") -> Dict:
        """成功返回"""
        return {
            "code": CODE_SUCCESS,
            "msg": msg,
            "data": data,
            "timestamp": int(time.time())
        }

    @staticmethod
    def fail(code: int, msg: str, data: Any = None) -> Dict:
        """自定义失败返回"""
        return {
            "code": code,
            "msg": msg,
            "data": data,
            "timestamp": int(time.time())
        }

    @staticmethod
    def param_err(msg: str = "请求参数错误") -> Dict:
        return ResponseHandler.fail(CODE_PARAM_ERROR, msg)

    @staticmethod
    def not_found(msg: str = "资源不存在") -> Dict:
        return ResponseHandler.fail(CODE_NOT_FOUND, msg)

    @staticmethod
    def upload_err(msg: str = "图片上传失败，请检查文件格式") -> Dict:
        return ResponseHandler.fail(CODE_UPLOAD_ERROR, msg)

    @staticmethod
    def model_err(msg: str = "模型推理异常") -> Dict:
        return ResponseHandler.fail(CODE_MODEL_ERROR, msg)

    @staticmethod
    def server_err(msg: str = "服务器内部错误") -> Dict:
        return ResponseHandler.fail(CODE_SERVER_ERROR, msg)

class GlobalExceptionCatcher:
    """全局异常捕获装饰器，用于API接口"""
    @staticmethod
    def catch(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException as he:
                logger.warning(f"业务异常：{he.detail}")
                return ResponseHandler.fail(he.status_code, he.detail)
            except Exception as e:
                err_stack = traceback.format_exc()
                logger.error(f"系统异常：{str(e)}\n堆栈信息：{err_stack}")
                return ResponseHandler.server_err(f"服务异常：{str(e)}")
        return wrapper

# 图片格式校验工具
def check_image_ext(filename: str) -> bool:
    allow_suffix = {"jpg", "jpeg", "png", "bmp"}
    suffix = filename.split(".")[-1].lower()
    return suffix in allow_suffix

# 文件大小校验
def check_file_size(file_bytes: bytes, max_mb: int = 10) -> bool:
    max_size = max_mb * 1024 * 1024
    return len(file_bytes) <= max_size