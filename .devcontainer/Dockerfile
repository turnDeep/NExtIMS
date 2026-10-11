FROM python:3.11-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir torch==2.9.1 --index-url ${TORCH_INDEX} \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt pytest
RUN useradd --create-home --uid 1000 devuser && chown -R devuser:devuser /opt/venv
WORKDIR /workspace
USER devuser
CMD ["bash"]
