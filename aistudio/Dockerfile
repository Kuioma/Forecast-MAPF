FROM final_mapf-gpt:0

# 配置时区
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8

ENV NVIDIA_VISIBLE_DEVICES=none NVIDIA_DRIVER_CAPABILITIES=all

# 容器内只有 /workspace 路径会持久化，其余路径下所有内容会随着容器回收而释放
WORKDIR /workspace

ARG PIP_INDEX_URL=https://mirrors.cowarobot.cn/simple
# 设置 pip 镜像源
ENV PIP_INDEX_URL=$PIP_INDEX_URL \
    PIP_TIMEOUT=100 \
    PIP_RETRIES=5


# clone 代码到 /tmp/flask 路径
# note: 代码路径与 /tmp/flask/run.sh 中python3 命令路径需要保持一致
# 若容器内已配置代理服务器且git.cowarobot.com 不在no_proxy 中，需要先取消代理才能拉取代码
# ENV http_proxy=  https_proxy= HTTP_PROXY= HTTPS_PROXY=
RUN rm -rf /tmp/aistudio && mkdir -p /tmp/aistudio && git clone https://git.cowarobot.com/yaoyongl/aistudio_dependence.git /tmp/aistudio
RUN pip3 install -r /tmp/aistudio/requirements.txt

# 缓存路径,必须为该路径
RUN mkdir /tmp/aistudio/cache && chmod 777 /tmp/aistudio/cache

# ssh 路径
RUN mkdir -p /run/sshd

# 安装code-server 【可选】若容器内非root用户 需 sudo dpkg -i code-server_4.9.1_amd64.deb
# RUN cd /tmp && wget http://ossapi.cowarobot.cn:30009/aistudio/lib/code-server_4.9.1_amd64.deb \
#     && dpkg -i code-server_4.9.1_amd64.deb \
#     && rm -rf code-server_4.9.1_amd64.deb

# 配置代理服务器 【可选】
ENV http_proxy=http://proxy.cowarobot.cn:8080 https_proxy=http://proxy.cowarobot.cn:8080
ENV no_proxy=localhost,127.0.0.1,.cowarobot.cn,172.16.1.235,172.17.1.235,172.17.1.5,172.16.110.100,git.cowarobot.com

CMD ["/tmp/aistudio/run.sh"]
