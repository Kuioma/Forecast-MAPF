from coops.train.coops import Coops, MODE_CUSTOM
import signal
import time

#基本配置
image="harbor.cowarobot.cn/aistudio/mapf-gpt:0"
queue_up=True #使用queue_up必须设置user和passwd
user="xiewende"
passwd="Cowa1230"

#选择训练模型
########################################################sudo
# tl stage1
worker=1
cpu_core=16
gpu_per_worker=0
gpu_kinds="4090"  # 3090,A6000,4090,6000Ada,L20
#cfg_file_path="config_files/camera_abnormal/classify-v8_mask.py"
#log_prefix="camera_abnormal-classify_v8-update_abnormal_classes"
########################################################

if __name__ == "__main__":
    retry_interval=3# s
    success=0
    signal.signal(signal.SIGINT, quit)
    signal.signal(signal.SIGTERM, quit)
    t0 = time.time()
    is_first_try = True
    while True:
        if success>0:
            break
        if is_first_try or time.time()-t0 > retry_interval:
            is_first_try=False
            try:
                Coops(
                    worker=worker,  # 节点数
                    cpu_core=cpu_core,  # 每节点的CPU核数
                    gpu_per_worker=gpu_per_worker,  # 每节点的GPU卡数
                    image=image,  # 训练容器镜像
                    mode=MODE_CUSTOM,  # 模式
                    gpu_kinds=[gpu_kinds],  # 3090,A6000,4090,6000Ada,L20
                    entry_path="aistudio.sh",  # 训练启动脚本相对路径
                    user=user,
                    passwd=passwd
                ).run()
                success+=1
            except Exception as e:
                print("err:",e)
            t0 = time.time()
        if not queue_up:
            break
        else:
            if not success:
                print(f"retry after {retry_interval - (time.time() - t0)} sec")
            else:
                break
            time.sleep(retry_interval - (time.time() - t0))
