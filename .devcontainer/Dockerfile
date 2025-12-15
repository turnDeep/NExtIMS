# Dockerfile for Mass Spectrum Prediction Dev Container
# RTX 50シリーズ（Blackwell sm_120）対応のCUDA 12.8環境

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 環境変数の設定（sm_120対応を明示）
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TORCH_CUDA_ARCH_LIST="9.0;12.0" \
    CUDA_LAUNCH_BLOCKING=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 作業ディレクトリ設定
WORKDIR /workspace

# システムパッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    # 基本ツール
    git wget curl vim build-essential cmake gcc g++ \
    ca-certificates gnupg lsb-release \
    # Python 3.11（PyTorch Nightlyと互換性が良い）
    python3.11 python3.11-dev python3.11-venv python3-pip \
    # 開発用依存関係
    libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev \
    # 化学計算用依存関係
    libopenblas-dev liblapack-dev libhdf5-dev \
    # Git LFS (BDE-db2ダウンロードに必要)
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Node.js のインストール (Claude CLI に必要)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Claude CLI のインストール
RUN npm install -g @anthropic-ai/claude-code

# Python 3.11をデフォルトに設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Python仮想環境の作成
RUN python3.11 -m venv /opt/venv

# システムのデフォルトPythonを仮想環境のPythonに強制的に置き換える
# これにより、いかなる状況でも仮想環境が使われることを保証する
RUN ln -sf /opt/venv/bin/python /usr/bin/python && \
    ln -sf /opt/venv/bin/python /usr/bin/python3 && \
    ln -sf /opt/venv/bin/pip /usr/bin/pip && \
    ln -sf /opt/venv/bin/pip /usr/bin/pip3

# 仮想環境のパスをPATHの先頭に追加
ENV PATH="/opt/venv/bin:$PATH"

# ===================================================
# RTX 50シリーズ対応: PyTorch Nightly (cu128) インストール
# ===================================================
# nvidia-nvshmem-cu12を先にインストールして依存関係の競合を回避
RUN pip install --no-cache-dir nvidia-nvshmem-cu12==3.4.5

# PyTorch nightlyをインストール
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# torchdata (BonDNet dependency) - Version 0.7.1 is compatible
RUN pip install --no-cache-dir torchdata==0.7.1

# ===================================================
# Pythonパッケージ一括インストール
# 依存関係の問題を回避するため、ビルドツールを先にインストール
# ===================================================
RUN pip install --no-cache-dir six hatchling wheel ninja

# 基本的な科学計算ライブラリ
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0 \
    pandas==2.2.2 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    plotly==5.20.0 \
    h5py==3.11.0 \
    pyyaml \
    tqdm \
    scikit-learn

# 化学構造処理ライブラリ
RUN pip install --no-cache-dir \
    rdkit \
    mordred \
    mol2vec

# Jupyter環境
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython

# ===================================================
# PyTorch Geometric と拡張ライブラリのインストール
# CUDA対応ビルド済みホイールを使用（sm_120対応）
# ===================================================
RUN pip install --no-cache-dir torch-geometric

# torch_scatterなど拡張ライブラリをソースからビルド
# sm_120 (RTX 5070 Ti) 対応のため、CUDA環境変数を設定してソースからビルド
# PyTorch Nightly (2.10.0+) はビルド済みホイールが存在しないため、直接ソースからビルド
# メモリ不足回避のため、ここでも念のため並列数を制限する環境変数を設定
RUN export FORCE_CUDA=1 && \
    export TORCH_CUDA_ARCH_LIST="9.0;12.0" && \
    export CUDA_HOME=/usr/local/cuda-12.8 && \
    export MAX_JOBS=2 && \
    echo "Building PyG extensions from source with CUDA support..." && \
    pip install --no-cache-dir --no-build-isolation torch-scatter && \
    pip install --no-cache-dir --no-build-isolation torch-sparse torch-cluster torch-spline-conv

# Open Graph Benchmark (OGB) for PCQM4Mv2 dataset
RUN pip install --no-cache-dir ogb>=1.3.6

# ===================================================
# DGL (Deep Graph Library) インストール
# BonDNetの依存関係として必要
# ===================================================
# PyTorch Nightlyと互換性を持たせるため、ソースからビルド
# sm_120 (RTX 5070 Ti) 対応のため、パッチを適用して手動ビルド
RUN export FORCE_CUDA=1 && \
    export TORCH_CUDA_ARCH_LIST="9.0;12.0" && \
    export CUDA_HOME=/usr/local/cuda-12.8 && \
    echo "Building DGL from source with CUDA support..." && \
    # 1. ソースコードを再帰的にクローン
    git clone --recursive https://github.com/dmlc/dgl.git /tmp/dgl_source && \
    cd /tmp/dgl_source && \
    # 2. パッチ適用: 単純にファイルの末尾にCUDA 12.8設定を追記する方式 (echoを使用)
    echo 'if (CUDA_VERSION VERSION_GREATER_EQUAL "12.8")' >> cmake/modules/CUDA.cmake && \
    echo '  list(APPEND dgl_known_gpu_archs "120")' >> cmake/modules/CUDA.cmake && \
    echo '  set(dgl_cuda_arch_ptx "120")' >> cmake/modules/CUDA.cmake && \
    echo 'endif()' >> cmake/modules/CUDA.cmake && \
    # 3. C++ライブラリのビルド
    mkdir build && cd build && \
    # Releaseビルドを指定し、CUDA_ARCH_BIN="12.0" (sm_120) のみをターゲットにする
    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DUSE_OPENMP=ON -DCUDA_ARCH_NAME=Manual -DCUDA_ARCH_BIN="12.0" .. && \
    # 【重要】メモリ不足回避のため、並列数を2に制限 (24GB RAM環境向け)
    make -j2 && \
    # 4. Pythonバインディングのインストール
    cd ../python && \
    pip install . && \
    # 5. クリーンアップ
    cd / && rm -rf /tmp/dgl_source

# ===================================================
# BonDNet インストール
# BDE (Bond Dissociation Energy) 計算に使用
# ===================================================
RUN pip install --no-cache-dir \
    git+https://github.com/mjwen/bondnet.git

# Git LFS の初期化 (BDE-db2データセット用)
RUN git lfs install

# 追加の機械学習ツール
RUN pip install --no-cache-dir \
    tensorboard \
    wandb \
    torch-ema

# 開発用ツール
RUN pip install --no-cache-dir \
    pytest \
    black \
    flake8 \
    mypy

# 非rootユーザーの作成
RUN useradd -m -s /bin/bash devuser && \
    chown -R devuser:devuser /workspace && \
    chown -R devuser:devuser /opt/venv

# bashrcの設定: ターミナル起動時に自動で仮想環境を有効化
RUN echo "source /opt/venv/bin/activate" >> /home/devuser/.bashrc && \
    echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Pythonパス設定
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# GPU検証スクリプトの作成
RUN cat <<'SCRIPT' > /usr/local/bin/verify-gpu.py
#!/usr/bin/env python3
import torch
import sys

print("=" * 60)
print("RTX 50シリーズ GPU検証")
print("=" * 60)

# CUDA利用可能性チェック
cuda_available = torch.cuda.is_available()
print(f"CUDA利用可能: {cuda_available}")

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f"GPUデバイス数: {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  メモリ: {props.total_memory / 1e9:.1f} GB")
        print(f"  SM数: {props.multi_processor_count}")

        # sm_120のチェック
        if props.major == 12 and props.minor == 0:
            print(f"  ✅ sm_120 (Blackwell) 検出!")

    # PyTorchバージョン情報
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")

    # 簡単なGPU演算テスト
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.mm(test_tensor, test_tensor)
        print("\n✅ GPU演算テスト成功!")
    except Exception as e:
        print(f"\n❌ GPU演算テスト失敗: {e}")
        sys.exit(1)

    # torch_scatter のCUDA対応確認
    try:
        import torch_scatter
        print(f"\n✅ torch_scatter インストール済み")
        # scatter演算テスト
        src = torch.randn(10, 5).cuda()
        index = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1]).cuda()
        out = torch_scatter.scatter(src, index, dim=0, reduce="sum")
        print(f"   torch_scatter CUDA演算テスト成功!")
    except Exception as e:
        print(f"\n❌ torch_scatter エラー: {e}")
        sys.exit(1)
else:
    print("❌ CUDAが利用できません")
    sys.exit(1)

print("=" * 60)
SCRIPT

RUN chmod +x /usr/local/bin/verify-gpu.py

USER devuser

# デフォルトコマンド
CMD ["/bin/bash"]
