FROM python:3.5.7-stretch

# install gcloud to make it easier to access cloud storage
RUN mkdir -p /usr/local/gcloud \
    && curl -q https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz -o /tmp/google-cloud-sdk.tar.gz \
    && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && rm /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh --usage-reporting false \
    && /usr/local/gcloud/google-cloud-sdk/bin/gcloud components install --quiet beta

ENV PATH /usr/local/gcloud/google-cloud-sdk/bin:$PATH

ENV PROJECT_FOLDER=/opt/sciencebeam-trainer-delft

WORKDIR ${PROJECT_FOLDER}

ENV VENV=/opt/venv
RUN python3 -m venv ${VENV} \
    && python3 -m venv ${VENV}
ENV VIRTUAL_ENV=${VENV} PYTHONUSERBASE=${VENV} PATH=${VENV}/bin:$PATH

COPY requirements.build.txt ./
RUN pip install -r requirements.build.txt

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY requirements.cpu.txt ./
RUN pip install -r requirements.cpu.txt

COPY requirements.delft.txt ./
RUN pip install -r requirements.delft.txt --no-deps

ARG install_dev
COPY requirements.dev.txt ./
RUN if [ "${install_dev}" = "y" ]; then pip install -r requirements.dev.txt; fi

COPY sciencebeam_trainer_delft ./sciencebeam_trainer_delft
COPY setup.py  MANIFEST.in ./

COPY config/embedding-registry.json ./

COPY .flake8 .pylintrc pytest.ini ./
COPY tests ./tests
