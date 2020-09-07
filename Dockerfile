FROM python:3.5.7-stretch

# install gcloud to make it easier to access cloud storage
RUN mkdir -p /usr/local/gcloud \
    && curl -q https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz -o /tmp/google-cloud-sdk.tar.gz \
    && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && rm /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh --usage-reporting false \
    && /usr/local/gcloud/google-cloud-sdk/bin/gcloud components install --quiet beta

ENV PATH /usr/local/gcloud/google-cloud-sdk/bin:$PATH

# install wapiti
ARG wapiti_source_download_url
RUN if [ ! -z "${wapiti_source_download_url}" ]; then \
    curl --location -q "${wapiti_source_download_url}" -o /tmp/wapiti.tar.gz \
    && ls -lh /tmp/wapiti.tar.gz \
    && mkdir -p /tmp/wapiti \
    && tar --strip-components 1 -C /tmp/wapiti -xvf /tmp/wapiti.tar.gz \
    && rm /tmp/wapiti.tar.gz \
    && cd /tmp/wapiti \
    && make \
    && make install \
    && rm -rf /tmp/wapiti; \
    fi

ENV PROJECT_FOLDER=/opt/sciencebeam-trainer-delft

WORKDIR ${PROJECT_FOLDER}

ENV PATH=/root/.local/bin:${PATH}

COPY requirements.build.txt ./
RUN pip install --user -r requirements.build.txt

COPY requirements.txt ./
RUN LMDB_FORCE_SYSTEM=1 \
    pip install --user -r requirements.txt

COPY requirements.cpu.txt ./
RUN pip install --user -r requirements.cpu.txt

COPY requirements.delft.txt ./
RUN pip install --user -r requirements.delft.txt --no-deps

ARG install_dev
COPY requirements.dev.txt ./
RUN if [ "${install_dev}" = "y" ]; then pip install -r requirements.dev.txt; fi

COPY sciencebeam_trainer_delft ./sciencebeam_trainer_delft
COPY setup.py  MANIFEST.in ./

COPY config/embedding-registry.json ./

COPY .flake8 .pylintrc pytest.ini ./
COPY tests ./tests

# add additional wrapper entrypoint for OVERRIDE_EMBEDDING_URL
COPY ./docker/entrypoint.sh ${PROJECT_FOLDER}/entrypoint.sh
ENTRYPOINT ["/opt/sciencebeam-trainer-delft/entrypoint.sh"]
CMD ["bash"]
