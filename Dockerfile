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
