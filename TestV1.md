# 📦 Installation (Ubuntu)

<details>
<summary>Install NVIDIA driver</summary>

</details>

<details>
<summary>Install environments with virtualenv</summary>

### Install python 3.10

```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.10
sudo apt-get install python3.10-dev
sudo apt-get install python3.10-distutils
sudo apt-get install python3.10-venv
```

### Create a virtual python environment
   - Make sure you already have `virtualenv`. Create a empty virtual environment for `anomalib`
   
        ```shell
        mkdir ~/venv
        cd ~/venv
        virtualenv -p python3.10 anomalib
        ```

### Activate the virtual python environment

- Go into the viurtual python environment
     ```shell
     source ~/venv/anomalib/bin/activate
     ```
- You would see:
     ```shell
     (anomalib) programer@programer:~/venv$ 
     ```
- **(optional)** If you want to quit from the virtual environment
     ```shell
     deactivate
     ```

- **(optional)**
     如果原本的virtualenv是python3.8安裝的,使用python3.10安裝的環境pip會有問題,須執行以下步驟：

     ```bash
     (anomalib) programer@programer:~/venv$ curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
     ```

### Install Anomalib from source

```bash
(anomalib) programer@programer:~/venv$  cd ~/xavier/xavier_git/anomalib
pip install -e .
```

</details>

<details>
<summary>Install environments with docker</summary>

See ***ci/README.md*** for more details.

```bash
sudo docker create --gpus all \
--shm-size=8G\
 -i -t \
 --mount type=bind,source=/home/xaviertung/xavier/xavier_git/anomalib/datasets,target=/home/user/datasets,readonly \
 --mount type=bind,source=/home/xaviertung/xavier/xavier_git/anomalib,target=/home/user/anomalib \
 -e TZ=Asia/Taipei \
 --name anomalib-ci-container anomalib-ci
```

</details>