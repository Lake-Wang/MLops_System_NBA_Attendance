# Try to place in a .ipynb on Chameleon
# Closely mimics labs
### Setting up AMD server
from chi import server, context, lease
import os, time
import yaml

context.version = "1.0" 
context.choose_project()
context.choose_site(default="CHI@TACC")

l = lease.get_lease(f"mltrain_project16") 
l.show()

username = os.getenv('USER') # all exp resources will have this prefix
s = server.Server(
    f"node-mltrain-{username}", 
    reservation_id=l.node_reservations[0]["id"],
    image_name="CC-Ubuntu24.04-hwe"
)
s.submit(idempotent=True)

s.associate_floating_ip()
s.refresh()
s.check_connectivity()
s.refresh()
s.show(type="widget")

# Clone GitHub
s.execute("git clone --recurse-submodules https://github.com/jasonmoon97/dynamic_nba_scheduling.git")

# Set up Docker
s.execute("curl -sSL https://get.docker.com/ | sudo sh")
s.execute("sudo groupadd -f docker; sudo usermod -aG docker $USER")

# Set AMD GPU server
s.execute("sudo apt update; wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb")
s.execute("sudo apt -y install ./amdgpu-install_6.3.60303-1_all.deb; sudo apt update")
s.execute("amdgpu-install -y --usecase=dkms")
s.execute("sudo apt -y install rocm-smi")
s.execute("sudo usermod -aG video,render $USER")
s.execute("sudo reboot")
time.sleep(30)
s.refresh()
s.check_connectivity()

s.execute("rocm-smi")
s.execute("sudo apt -y install cmake libncurses-dev libsystemd-dev libudev-dev libdrm-dev libgtest-dev")
s.execute("git clone https://github.com/Syllo/nvtop")
s.execute("mkdir -p nvtop/build && cd nvtop/build && cmake .. -DAMDGPU_SUPPORT=ON && sudo make install")

# Build image - MLFlow container
s.execute("docker build -t jupyter-mlflow -f dynamic_nba_scheduling/ml-train/docker/Dockerfile.jupyter-torch-mlflow-rocm .")

## Prepare data - fetch from block storage
s.execute("curl https://rclone.org/install.sh | sudo bash")
s.execute("sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf")

s.execute("mkdir -p ~/.config/rclone")

config_path = os.path.expanduser("~/.config/rclone/rclone.conf")

with open("clouds.yaml", "r") as file:
    secret = yaml.safe_load(file)

cloud_config = secret["clouds"]["chi_tacc"]
auth = cloud_config["auth"]

application_credential_id = auth["application_credential_id"]
application_credential_secret = auth["application_credential_secret"]

rclone_configs = f"""
[chi_tacc]
type = swift
user_id = 12c0ee0dd863e5fc52f1cb58899047dc431eba2bceb29f15984a05bf9c0bba8f
application_credential_id = {application_credential_id}
application_credential_secret = {application_credential_secret}
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
"""

safe_rclone = rclone_configs.strip()

s.execute(f""" cat <<EOF > ~/.config/rclone/rclone.conf
{safe_rclone}
EOF
""")

s.execute("sudo mkdir -p /mnt/object")
s.execute("sudo chown -R cc /mnt/object")
s.execute("sudo chgrp -R cc /mnt/object")
s.execute("rclone mount chi_tacc:object-persist-project16 /mnt/object --read-only --allow-other --daemon")
s.execute("ls /mnt/object")

## MLFlow tracking server
s.execute("docker compose -f dynamic_nba_scheduling/ml-train/docker/docker-compose-mlflow.yaml up -d")

# Jupyter container
s.execute("""HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )
docker run  -d --rm  -p 8888:8888 \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add $(getent group | grep render | cut -d':' -f 3) \
    --shm-size 16G \
    -v ~/dynamic_nba_scheduling/ml-train:/home/jovyan/work/ \
    -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
    -e NBA_DATA_DIR=/mnt/nba_data \
    --mount type=bind,source=/mnt/object,target=/mnt/nba_data,readonly\
    --name jupyter \
    jupyter-mlflow""")

# Train models (in Jupyter container)
s.execute("python3 dynamic_nba_scheduling/ml-train/train_model1.py")
s.execute("python3 dynamic_nba_scheduling/ml-train/train_model2.py")

## Start up the Ray server
s.execute("docker build -t ray-rocm:2.42.1 -f dynamic_nba_scheduling/ml-train/docker/Dockerfile.ray-rocm .")
s.execute("export HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )")
s.execute("docker compose -f dynamic_nba_scheduling/ml-train/docker/docker-compose-ray-rocm.yaml up -d")
s.execute("docker build -t jupyter-ray -f dynamic_nba_scheduling/ml-train/docker/Dockerfile.jupyter-ray .")
s.execute("HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4 )")
s.execute("""docker run  -d --rm  -p 8888:8888 \
    -v ~/dynamic_nba_scheduling/ml-train/workspace_ray:/home/jovyan/work/ \
    -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
    --name jupyter \
    jupyter-ray""")

## Submit Ray job (in workspace_ray)
s.execute("ray job submit --runtime-env runtime.json --entrypoint-num-gpus 1 --entrypoint-num-cpus 8 --verbose  --working-dir .  -- python dynamic_nba_scheduling/ml-train/train_model1.py")
s.execute("ray job submit --runtime-env runtime.json --entrypoint-num-gpus 1 --entrypoint-num-cpus 8 --verbose  --working-dir .  -- python dynamic_nba_scheduling/ml-train/train_model2.py")