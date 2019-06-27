FROM python:3.6.8-stretch

ARG delft_repo=kermitt2/delft
ARG delft_tag=master

RUN curl --progress-bar --location \
  "https://github.com/${delft_repo}/archive/${delft_tag}.tar.gz" \
  --output "/tmp/${delft_tag}.tar.gz" \
  && tar -C "/opt" -xvf "/tmp/${delft_tag}.tar.gz" \
  && rm "/tmp/${delft_tag}.tar.gz" \
  && ln -s "/opt/delft-${delft_tag}" "/opt/delft"

WORKDIR /opt/delft

ENV PATH=/root/.local/bin:${PATH}
RUN pip install --user -r requirements.txt
RUN pip install --user -r requirements.cpu.txt

RUN mkdir -p data/db/glove-6B-50d \
  && curl --progress-bar --location \
  "https://github.com/elifesciences/sciencebeam-models/releases/download/v0.0.1/glove-6B-50d.mdb.gz" \
  | gunzip - \
  > "data/db/glove-6B-50d/data.mdb"

ARG install_dev
COPY requirements.dev.txt /opt/sciencebeam-trainer-delft/
RUN if [ "${install_dev}" = "y" ]; then pip install -r /opt/sciencebeam-trainer-delft/requirements.dev.txt; fi

COPY sciencebeam_trainer_delft ./sciencebeam_trainer_delft

COPY .flake8 .pylintrc ./
