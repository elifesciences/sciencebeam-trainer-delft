FROM python:3.7.3-stretch

ARG delft_repo=kermitt2/delft
ARG delft_tag=master

RUN curl --progress-bar --location \
  "https://github.com/${delft_repo}/archive/${delft_tag}.tar.gz" \
  --output "/tmp/${delft_tag}.tar.gz" \
  && tar -C "/opt" -xvf "/tmp/${delft_tag}.tar.gz" \
  && rm "/tmp/${delft_tag}.tar.gz" \
  && ln -s "/opt/delft-${delft_tag}" "/opt/delft"
