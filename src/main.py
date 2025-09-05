import uvicorn
from src.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    config = uvicorn.Config(
        "src.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        limit_max_request_size=settings.MAX_FILE_MB * 1024 * 1024, 
    )
    server = uvicorn.Server(config)
    server.run()


