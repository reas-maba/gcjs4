import uvicorn
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.core.recognizer import DogRecognizer
from backend.utils.data_loader import load_all_breed_info
import logging

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DogRec")

# 初始化应用
app = FastAPI(title="犬种智能识别系统 V1.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局加载识别模型（仅初始化一次）
recognizer = DogRecognizer()
breed_info_map = load_all_breed_info()

@app.get("/")
async def root():
    return {
        "msg": "犬种识别API服务运行正常",
        "api_list": {
            "POST /predict": "上传图片识别犬种",
            "GET /breeds": "获取全部犬种名称",
            "GET /breed/detail": "单犬种详情",
            "GET /breeds/all": "全部犬种完整资料"
        }
    }

@app.post("/predict")
async def predict_dog(file: UploadFile = File(...)):
    try:
        logger.info("接收图片识别请求")
        img_bytes = await file.read()
        result = recognizer.get_top5_prediction(img_bytes)
        return {"code": 200, "data": result}
    except Exception as e:
        logger.error(f"识别失败: {str(e)}")
        return {"code": 500, "msg": f"识别异常：{str(e)}"}

@app.get("/breeds")
async def get_breed_list():
    breed_list = list(breed_info_map.keys())
    return {"code": 200, "breeds": breed_list}

@app.get("/breed/detail")
async def get_single_breed(breed: str = Query(..., description="犬种中文名")):
    if breed not in breed_info_map:
        return {"code": 404, "msg": "未查询到该犬种信息"}
    return {"code": 200, "data": breed_info_map[breed]}

@app.get("/breeds/all")
async def get_all_breed_info():
    return {"code": 200, "data": breed_info_map}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False)