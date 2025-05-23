# CUDAとPython 3.9を含むベースイメージ
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive

# プロキシ環境変数の引継ぎ
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=${HTTP_PROXY} 
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    sudo \
    tmux \
    git \
    wget \
    unzip \
    libgomp1 \
    swig \
    netcat-openbsd \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# # シンボリックリンクを作成
# RUN ln -s /usr/bin/python3 /usr/bin/python && \
#     ln -s /usr/bin/pip3 /usr/bin/pip

# apt proxyの設定
RUN if [ -n "${HTTP_PROXY}" ]; then \
    echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" > /etc/apt/apt.conf.d/proxy.conf; \
    fi

# 作業ディレクトリを設定
WORKDIR /workspace

# ボリュームを定義（必要に応じて）
VOLUME /workspace/data

# フォントインストール
RUN wget https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip -O /tmp/fonts_noto.zip && \
    mkdir -p /usr/share/fonts &&\
    unzip /tmp/fonts_noto.zip -d /usr/share/fonts

# ユーザー設定
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# requirements.txtのコピー
COPY requirements.txt .

# Python依存関係のインストール
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# PYTHONPATHの設定
ENV PYTHONPATH=/workspace

# 非ルートユーザーに切り替え
USER $USERNAME

# デフォルトコマンド
CMD ["/bin/bash"]