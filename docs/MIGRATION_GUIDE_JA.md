# Docker DesktopからWSL2ネイティブDockerへの移行ガイド

## 概要
現在、Windows上のDocker Desktopを利用してWSL2環境で開発を行っていますが、これをDocker DesktopからWSL2ネイティブのDocker Engineに移行する手順を説明します。これにより、WSL2内でのパフォーマンス向上や、よりLinuxネイティブな動作環境が期待できます。

## Step 1: 【準備】現状のバックアップ（必須）

Docker Desktopをアンインストールすると、コンテナ、イメージ、ボリューム（データ）はすべて削除されます。特に**Ollamaのモデルデータ（数GB～数十GB）**や**OpenWebUIの会話履歴**はボリュームに保存されているため、必ずバックアップが必要です。

WindowsのPowerShellまたはコマンドプロンプトで以下の操作を行ってください（出力先をWindows側にすると管理しやすいため）。

### 1. イメージの書き出し (docker save)
必要なDockerイメージをファイル（.tar）として保存します。

```powershell
# 保存先フォルダを作成（例: C:\docker_backup）
mkdir C:\docker_backup
cd C:\docker_backup

# イメージ一覧を確認
docker images

# イメージを書き出し (例: ollama/ollama と ghcr.io/open-webui/open-webui:main)
docker save -o ollama_image.tar ollama/ollama:latest
docker save -o openwebui_image.tar ghcr.io/open-webui/open-webui:main

# NExtIMSなどの自作イメージがあればそれも
docker save -o nextims_image.tar nextims:latest
```

### 2. ボリュームの書き出し (重要)
Ollamaのモデルデータなどが含まれるボリュームをバックアップします。

まず、どのボリュームが使われているか確認します。
```powershell
docker volume ls
```

**Ollamaのボリュームバックアップ:**
```powershell
# ollamaという名前のボリュームを backup_ollama.tar に保存
docker run --rm -v ollama:/volume -v ${PWD}:/backup ubuntu tar cvf /backup/backup_ollama.tar /volume
```

**OpenWebUIのボリュームバックアップ:**
```powershell
# open-webuiという名前のボリュームを backup_openwebui.tar に保存
docker run --rm -v open-webui:/volume -v ${PWD}:/backup ubuntu tar cvf /backup/backup_openwebui.tar /volume
```
※ `ollama` や `open-webui` の部分は、`docker volume ls` で確認した実際のボリューム名に置き換えてください。

---

## Step 2: 【削除】Docker Desktopのアンインストール

1.  Docker Desktopを終了します。
2.  Windowsの「設定」>「アプリ」から **Docker Desktop** をアンインストールします。
3.  PCを再起動します。

---

## Step 3: 【導入】WSL2 (Ubuntu) へのインストール

ここからは **WSL2のターミナル（Ubuntu）** を開いて作業します。

### 1. 古いDocker関連の削除 & リポジトリ追加
```bash
# 競合を防ぐため古いパッケージを削除
sudo apt-get remove docker docker-engine docker.io containerd runc

# 必要なパッケージのインストール
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Dockerの公式GPGキーを追加
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# リポジトリのセットアップ
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 2. Docker Engineのインストール
```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3. ユーザー権限の設定（sudoなしでdockerコマンドを使えるようにする）
```bash
sudo usermod -aG docker $USER
# 設定を反映させるため、一度WSLを閉じて開き直すか、以下を実行
newgrp docker
```
動作確認: `docker ps` を実行してエラーが出なければOKです。

### 4. NVIDIA Container Toolkitのインストール（AI/GPU利用に必須）
NExtIMSやOllamaでGPUを使うために必要です。

```bash
# リポジトリ設定
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# インストール
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 設定の生成とDocker再起動
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
動作確認: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` を実行し、GPU情報が表示されれば成功です。

---

## Step 4: 【復元】データのリストア

Windows側に保存したバックアップファイル（C:\docker_backup）を使って復元します。WSLからWindowsのファイルには `/mnt/c/` でアクセスできます。

### 1. イメージの読み込み
```bash
cd /mnt/c/docker_backup

docker load -i ollama_image.tar
docker load -i openwebui_image.tar
# その他保存したイメージがあれば同様に
```

### 2. ボリュームの復元
まず空のボリュームを作成し、そこにtarを展開します。

**Ollamaの復元:**
```bash
# ボリューム作成
docker volume create ollama

# データを展開 (tarの中身が /volume ディレクトリ以下に展開されるようにする)
# --strip-components=1 はバックアップ時のディレクトリ構成に応じて調整してください
docker run --rm -v ollama:/volume -v /mnt/c/docker_backup:/backup ubuntu bash -c "cd /volume && tar xvf /backup/backup_ollama.tar --strip-components=1"
```

**OpenWebUIの復元:**
```bash
docker volume create open-webui
docker run --rm -v open-webui:/volume -v /mnt/c/docker_backup:/backup ubuntu bash -c "cd /volume && tar xvf /backup/backup_openwebui.tar --strip-components=1"
```

### 3. コンテナの起動 (リストアの完了)
イメージとボリュームの準備ができたら、最後にコンテナを起動して完了です。
`docker-compose.yml` ファイルがあるディレクトリ（例: `~/projects/ollama-openwebui`）に移動し、以下のコマンドを実行します。

```bash
cd ~/projects/ollama-openwebui  # 実際のパスに合わせてください
docker compose up -d
```
これにより、復元したイメージとボリュームを使ってコンテナが立ち上がります。もし `docker-compose.yml` がない場合は、`docker run` コマンドでボリュームをマウントして起動してください。

---

## Step 5: VSCodeと各リポジトリの対応

### VSCodeの設定
1.  VSCodeに拡張機能 **「WSL」** と **「Dev Containers」** がインストールされていることを確認します。
2.  VSCode左下の緑色のボタンをクリックし、「New Window using Distro...」等でWSL (Ubuntu) に接続します。
3.  以後は「Remote - WSL」の中で作業します。`docker` コマンドがVSCode内のターミナルで利用可能です。

### 各リポジトリ別の注意点

1.  **NExtIMS (現在のリポジトリ)**
    *   **ファイルの移動**: ソースコードがWindows側（`/mnt/c/...`）にある場合は、WSLのホームディレクトリ（`~/projects/NExtIMS` など）に移動することを強く推奨します（ディスクアクセス速度向上のため）。
    *   **Dev Container**: `.devcontainer/devcontainer.json` があれば、VSCodeでフォルダを開いた際に「Reopen in Container」と表示され、そのまま利用可能です。

2.  **Computational-Chemistry-AI**
    *   同様にファイルをWSL領域（`~/`以下）に移動してください。
    *   GPUを使用する場合、`docker-compose.yml` に以下の記述があるか確認してください。
        ```yaml
        services:
          service-name:
            deploy:
              resources:
                reservations:
                  devices:
                    - driver: nvidia
                      count: 1
                      capabilities: [gpu]
        ```

3.  **Ollama & Open WebUI**
    *   `docker-compose.yml` があるディレクトリに移動し、`docker compose up -d` で起動します。
    *   ボリューム（`ollama`, `open-webui`）が正しく復元・マウントされていれば、以前ダウンロードしたLLMモデルやチャット履歴がそのまま利用可能です。
    *   GPUが認識されているか確認するには、Ollama起動中にログを確認するか、`docker exec -it <ollamaのコンテナID> nvidia-smi` を実行してください。

---

## Windowsブラウザからのアクセスについて

Docker Engine on WSL2で起動したコンテナ（OllamaやOpenWebUIなど）は、**Windows側のブラウザから `localhost` 経由でそのままアクセス可能**です。

*   例えば、OpenWebUIがポート `3000` で動いている場合、WindowsのChromeやEdgeで `http://localhost:3000` にアクセスすれば開けます。
*   WSL2が自動的にポート転送を行うため、特別な設定は不要です。
*   もしアクセスできない場合は、以下の点を確認してください：
    1.  コンテナが正しく起動しているか (`docker ps`)
    2.  `docker-compose.yml` で `ports` の設定が正しいか（例: `"3000:8080"`）
    3.  Windowsのファイアウォール設定（稀にブロックされることがありますが、通常は自動的に許可されます）

---

## オフライン環境でのインストール手順

インターネットに接続できないPC（オフラインPC）にインストールする場合、**インターネットに接続できる別のPC（オンラインPC）でインストーラー（.debファイル）をダウンロードし、USBメモリ等でオフラインPCに移動させる**必要があります。

ここでは、**オンラインPC上でWSL2 (Ubuntu) を使って必要なファイルをダウンロードする方法**を説明します。
（オンラインPCがWindowsのみの場合でも、WSL2をインストールして行うのが最も確実です）

### Step A: オンラインPCでのファイルダウンロード

オンラインPCのWSL2ターミナルで以下のコマンドを実行し、必要なパッケージをすべてダウンロードします。

#### 1. 準備
作業用ディレクトリを作成します。
```bash
mkdir -p ~/docker-offline
cd ~/docker-offline
```

#### 2. Docker Engine関連のダウンロード
Docker公式リポジトリを追加してからダウンロードします。
```bash
# GPGキーとリポジトリの追加（まだの場合）
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

# パッケージのダウンロード
# docker-ce, docker-ce-cli, containerd.io, docker-buildx-plugin, docker-compose-plugin
apt-get download docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### 3. NVIDIA Container Toolkit関連のダウンロード
NVIDIA公式リポジトリを追加してからダウンロードします。
```bash
# リポジトリ設定（まだの場合）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update

# パッケージのダウンロード
# nvidia-container-toolkit, nvidia-container-toolkit-base, libnvidia-container-tools, libnvidia-container1
apt-get download nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1
```
※ 依存関係のエラーが出る場合は、`apt-cache depends <パッケージ名>` で不足している依存パッケージを確認し、それらも `apt-get download` してください。

### Step B: ファイルの移動

`~/docker-offline` ディレクトリにダウンロードされた `.deb` ファイル群（合計10個程度）と、Step 1で作成したバックアップ（イメージとボリュームの `.tar` ファイル）をUSBメモリなどにコピーします。

Windows側からWSL2のファイルにアクセスするには、エクスプローラーのアドレスバーに `\\wsl$\Ubuntu\home\ユーザー名\docker-offline` と入力します。

### Step C: オフラインPCでのインストール

オフラインPCのWSL2にファイルを移動し、インストールを行います。

1.  ファイルをオフラインPCのWSL2上の適当なフォルダ（例: `~/install-files`）にコピーします。
2.  そのディレクトリに移動してインストールを実行します。

```bash
cd ~/install-files
sudo dpkg -i *.deb
```

3.  もし依存関係のエラーが出た場合（例: `libc6` や `iptables` など標準ライブラリのバージョン違いなど）、エラーメッセージを確認して不足しているパッケージをオンラインPCで追加ダウンロードしてくる必要があります。
4.  インストール完了後、NVIDIAの設定を行います。
    ```bash
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```
5.  最後に、Step 4の手順に従ってイメージとボリュームをリストアしてください。

---

## オフライン環境でのOllamaモデル追加 (GGUF)

Hugging Face等からダウンロードしたGGUFファイル（`.gguf`）をオフライン環境のOllamaに追加する手順です。
例として `gemma-3-27b-it` を使用します。

### Step 1: モデルファイルのダウンロード (オンラインPC)

オンラインPCで、以下のURLなどからGGUFファイルをダウンロードします。
URL: https://huggingface.co/unsloth/gemma-3-27b-it-GGUF

**GPU VRAMに応じたファイルの選択目安:**
*   **VRAM 24GB以上** (RTX 3090/4090等): `Q4_K_M` (約16.5GB) が推奨です。品質と速度のバランスが良いです。
*   **VRAM 16GB以下**: `Q4_K_M` は動作が厳しい（またはコンテキストサイズが取れない）可能性があります。`Q3_K_M` や `Q2_K` を検討するか、一部CPUオフロードを許容する必要があります。

ダウンロードした `.gguf` ファイルをUSBメモリ等でオフラインPCのWSL2へ移動させてください（例: `~/models/gemma-3-27b-it-Q4_K_M.gguf`）。

### Step 2: Modelfileの作成 (オフラインPC)

WSL2上で、モデルと同じディレクトリに `Modelfile` という名前のテキストファイルを作成します。

```bash
cd ~/models
nano Modelfile
```

中身は以下の一行だけでOKです。**Dockerコンテナ内のパスを指定するため、`/root/` を付けます。**
```dockerfile
FROM /root/gemma-3-27b-it-Q4_K_M.gguf
```
※ ファイル名は実際にダウンロードしたものに合わせてください。

### Step 3: Ollamaコンテナへの取り込み

OllamaはDockerコンテナ内で動作しているため、ファイルをコンテナ内に転送してからコマンドを実行します。

1.  **コンテナ名の確認**
    ```bash
    docker ps
    ```
    `IMAGE` が `ollama/ollama` となっているコンテナの `NAMES` を確認します（例: `ollama`）。

2.  **ファイルの転送**
    GGUFファイルとModelfileをコンテナ内の `/root` ディレクトリにコピーします。
    ```bash
    # docker cp <ローカルファイルパス> <コンテナ名>:<コンテナ内パス>
    # (例: コンテナ名が "ollama" の場合)
    docker cp gemma-3-27b-it-Q4_K_M.gguf ollama:/root/
    docker cp Modelfile ollama:/root/
    ```

3.  **モデルの作成実行**
    コンテナ内で `ollama create` コマンドを実行します。
    ```bash
    docker exec -it ollama ollama create gemma-3-27b -f /root/Modelfile
    ```
    処理が完了すると `success` と表示されます。

4.  **確認**
    ```bash
    docker exec -it ollama ollama list
    ```

5.  **後片付け（推奨）**
    コンテナ内にコピーしたGGUFファイルは、取り込み完了後は不要になるため削除してディスク容量を空けます。
    ```bash
    docker exec -it ollama rm /root/gemma-3-27b-it-Q4_K_M.gguf /root/Modelfile
    ```

これで、`ollama run gemma-3-27b` や Open WebUI からこのモデルを選択できるようになります。

---

## オフライン環境でのVSCode & Dev Containers導入

オフラインPCでVSCodeの「Remote - WSL」や「Dev Containers」機能を使用するには、VSCode本体と拡張機能に加え、通常は自動ダウンロードされる**VS Code Server**を手動で導入する必要があります。

### Step 1: 必要なファイルのダウンロード (オンラインPC)

オンラインPCで以下のファイルを準備します。

1.  **VSCode本体 (Windows版)**
    公式サイトから `System Installer` または `User Installer` (x64) をダウンロードします。

2.  **拡張ファイル (.vsix)**
    Marketplaceの各ページ右側にある「Download Extension」リンクからダウンロードします。
    *   **WSL**: `ms-vscode-remote.remote-wsl`
    *   **Dev Containers**: `ms-vscode-remote.remote-containers`
    *   **Docker**: `ms-azuretools.vscode-docker`

3.  **VS Code Server (重要)**
    VSCodeがWSLに接続する際に必要なサーバープログラムです。これは**使用するVSCodeのバージョン（Commit ID）と完全に一致する必要があります**。

    *   **Commit IDの確認方法**:
        オンラインPCにダウンロードしたVSCodeを一時的にインストールするか、インストーラーのプロパティ詳細等からバージョンを確認し、Web検索で「VSCode <バージョン> commit id」と検索して特定します。（または、オフラインPCにVSCodeをインストールした後、ヘルプ > バージョン情報 から確認します）

    *   **ダウンロード**:
        以下のURLの `<COMMIT_ID>` 部分を実際のID（40桁の英数字）に置き換えてダウンロードします。
        `https://update.code.visualstudio.com/commit:<COMMIT_ID>/server-linux-x64/stable`
        ファイル名: `vscode-server-linux-x64.tar.gz`

### Step 2: インストールと配置 (オフラインPC)

ファイルをオフラインPCに移動し、以下の手順でセットアップします。

1.  **VSCodeと拡張機能のインストール**
    *   VSCode本体をインストールします。
    *   コマンドプロンプト等で拡張機能をインストールします（またはVSCode上の「VSIXからインストール」メニューを使用）。
        ```powershell
        code --install-extension ms-vscode-remote.remote-wsl.vsix
        code --install-extension ms-vscode-remote.remote-containers.vsix
        code --install-extension ms-azuretools.vscode-docker.vsix
        ```

2.  **VS Code Serverの手動配置 (WSL側)**
    WSL2 (Ubuntu) を起動し、ダウンロードした `vscode-server-linux-x64.tar.gz` を以下の場所に展開します。

    ```bash
    # Commit IDを変数にセット (例: a1b2c3...)
    COMMIT_ID="<確認したCommit ID>"

    # ディレクトリ作成
    mkdir -p ~/.vscode-server/bin/${COMMIT_ID}

    # 展開 (tar.gzがある場所で実行)
    tar -xzf vscode-server-linux-x64.tar.gz -C ~/.vscode-server/bin/${COMMIT_ID} --strip-components=1

    # インストール完了マーカーの作成
    touch ~/.vscode-server/bin/${COMMIT_ID}/0
    ```

3.  **接続確認**
    VSCodeを開き、左下の緑色のボタンから「New Window using Distro...」を選択してWSLに接続できるか確認します。
    接続できれば、Dev Containersも利用可能です。
