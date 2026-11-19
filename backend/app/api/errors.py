import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError

logger = logging.getLogger(__name__)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Log an AppError once and convert it to a JSON error response."""
    extra = {
        "event": "application_error",
        "error_code": exc.code,
        "status_code": exc.status_code,
        "path": request.url.path,
        "context": exc.context,
    }
    if exc.status_code >= 500:
        logger.error(exc.message, extra=extra, exc_info=exc)
    else:
        logger.warning(exc.message, extra=extra)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
