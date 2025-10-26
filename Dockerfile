# Dockerfile
FROM python:3.12-slim AS base

ARG APP_HOME=/app
ENV APP_HOME=${APP_HOME}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (build-essential often needed for scientific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create a non-root user and own the app dir
RUN useradd -m appuser && chown -R appuser:appuser ${APP_HOME}
USER appuser

# Where SQLite DB & collected static will live
RUN mkdir -p ${APP_HOME}/data
VOLUME ["${APP_HOME}/data"]

# Gunicorn defaults
ENV PORT=8000
ENV DJANGO_SETTINGS_MODULE=project.settings  # <-- change to your settings module

# Expose HTTP port
EXPOSE 8000

# Entrypoint runs migrations + collectstatic, then execs CMD
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Use gunicorn in container; swap to runserver for quick dev if you want
CMD ["gunicorn", "project.wsgi:application", \
     "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120"]
