import os,logging,uvicorn
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("main")
APP_HOST=os.getenv("APP_HOST","0.0.0.0")
APP_PORT=int(os.getenv("APP_PORT","8000"))
if __name__=="__main__":
    logger.info("Starting uvicorn (dev) on %s:%d",APP_HOST,APP_PORT)
    uvicorn.run("inference_service:app",host=APP_HOST,port=APP_PORT,log_level=os.getenv("LOG_LEVEL","info"))
