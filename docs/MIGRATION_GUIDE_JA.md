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
