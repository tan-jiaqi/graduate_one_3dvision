FROM ubuntu:jammy-20240427

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# install ca-cerfication
COPY libssl1.1_1.1.1f-1ubuntu2_amd64.deb openssl_1.1.1f-1ubuntu2_amd64.deb ca-certificates_20240203_all.deb /tmp/
RUN dpkg -i /tmp/libssl1.1_1.1.1f-1ubuntu2_amd64.deb && dpkg -i /tmp/openssl_1.1.1f-1ubuntu2_amd64.deb && dpkg -i /tmp/ca-certificates_20240203_all.deb && rm -rf /tmp/*.deb

# use tsinghua apt mirror
COPY sources.list /etc/apt/sources.list

# Install some basic utilities.
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends \
  curl \
  sudo \
  bzip2 \
  wget \
  git \
  vim \
  p7zip-full

# set timezone to Shanghai
RUN printf "tzdata tzdata/Areas select Asia\ntzdata tzdata/Zones/Asia select Shanghai\n" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
  && rm -rf /var/lib/apt/lists/*

# import cuda apt source
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
# use nvidia chinese mirror
COPY cuda-ubuntu2204-x86_64.list /etc/apt/sources.list.d/


# Create a working directory.
RUN mkdir /workspace
WORKDIR /workspace

# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/zsh user \
  && chown -R user:user /workspace
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# copy required file for build images
COPY deep3d_file deep3d_file

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
  && chmod -R 777 $HOME

# Default powerline10k theme, 4 plugins installed
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
  -p z -p sudo \
  -p https://github.com/zsh-users/zsh-syntax-highlighting \
  -p https://github.com/zsh-users/zsh-autosuggestions

# choose zsh to be the default terminal
RUN sudo chsh -s $(which zsh)

# Download and install Micromamba.
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
  | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
  MAMBA_ROOT_PREFIX=/home/user/micromamba \
  CONDA_PREFIX=/home/user/micromamba \
  PATH=/home/user/micromamba/bin:$PATH

# use conda chinese mirror； import vimrc and p10k configuration
COPY .vimrc .p10k.zsh $HOME/

# create python env and install required libraries
RUN micromamba create -qy -n base python=3.10 -c conda-forge \
  && micromamba activate base \
  && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
  && pip install wandb matplotlib numpy \
  && micromamba clean -qya

# use pip chinese mirror
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# activate base env
RUN echo "micromamba activate base" >> ~/.zshrc

# add p10k into .zshrc
RUN echo "# To customize prompt, run `p10k configure` or edit ~/.p10k.zsh." >> ~/.zshrc
RUN echo "[[ ! -f ~/.p10k.zsh ]] || source ~/.p10k.zsh" >> ~/.zshrc

CMD [ "/bin/zsh" ]

RUN sudo chmod -R 777 ./FunctionFitFinal
RUN cd FunctionFitFinal && sudo chmod -R 777 ./predictions
