FROM python:3.11-slim

# Make Python friendlier in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Stage: bring in whatever the build context is (repo root or /core)
WORKDIR /build
COPY . /build

# Install deps from the right requirements file:
# - If the context is repo root, use /build/core/requirements.txt
# - If the context is /core, fall back to /build/requirements.txt
RUN set -eux; \
    if [ -f /build/core/requirements.txt ]; then REQ=/build/core/requirements.txt; else REQ=/build/requirements.txt; fi; \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r "$REQ"

# Copy app code to /app from the correct place:
# - If repo root context → copy /build/core/*
# - If /core context     → copy /build/*
RUN set -eux; \
    rm -rf /app && mkdir -p /app; \
    if [ -d /build/core ]; then cp -R /build/core/. /app/; else cp -R /build/. /app/; fi

WORKDIR /app
# Port for uvicorn
ENV PORT=8000
EXPOSE 8000

# Start (ASCII hyphens only!)
CMD ["sh","-lc","python -m uvicorn app:app --host :: --port ${PORT:-8000} --log-level info"]



