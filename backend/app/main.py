from fastapi import FastAPI

from app.api.errors import app_error_handler
from app.api.health import router as health_router
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    # Aligning log level intelligently based on T1.3 debug configuration status
    log_level = "DEBUG" if settings.debug else "INFO"
    configure_logging(log_level)

    application = FastAPI(
        title=settings.app_name,
        description="Ask questions about public GitHub repositories.",
        version="0.1.0",
    )
    application.add_exception_handler(AppError, app_error_handler)
    application.include_router(health_router)
    return application


app = create_app()
