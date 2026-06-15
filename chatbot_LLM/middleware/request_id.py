"""
middleware/request_id.py
========================
Assigns a unique request ID to every incoming request.

The ID is taken from the X-Request-ID header if provided
by the caller (e.g. the Node.js backend), otherwise a new
UUID is generated. The ID is added to the response headers
so it can be traced end to end across services.
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestIDMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that assigns X-Request-ID to every request."""

    async def dispatch(self, request: Request, call_next):
        request_id = (
            request.headers.get("x-request-id") or str(uuid.uuid4())
        )
        # Attach to request state so handlers can access it
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
