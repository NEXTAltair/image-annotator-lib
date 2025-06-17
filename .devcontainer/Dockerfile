FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python packages
RUN pip install --upgrade pip \
    && pip install black pylint ruff

# Set the default user
USER $USERNAME

# Set working directory
WORKDIR /workspace

# Add uv to PATH
ENV PATH="/home/${USERNAME}/.cargo/bin:${PATH}" 