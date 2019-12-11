FROM continuumio/miniconda3:4.7.12
LABEL maintainer="bn<at>benbenbang.io"
LABEL version="1.0"

RUN touch .dockerenv && \
    mkdir -p ./workplace/lib

WORKDIR ./workplace
COPY ./requirements.txt ./requirements.txt
COPY ./src/benchmark.py ./benchmark.py
COPY ./src/libs/__init__.py ./libs/__init__.py
COPY ./src/libs/params.py ./libs/params.py

RUN apt-get update && \
    apt-get install -y locales locales-all libgl1-mesa-glx libboost-all-dev cmake wget \
    libglib2.0-0 libxext6 libsm6 libxrender1 gcc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

RUN conda update --all && \
    pip install -r requirements.txt && \
    pip install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "python", "benchmark.py" ]
