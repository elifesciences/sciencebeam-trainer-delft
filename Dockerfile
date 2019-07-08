FROM python:3.6.8-stretch

ENV PROJECT_FOLDER=/opt/sciencebeam-trainer-delft

WORKDIR ${PROJECT_FOLDER}

ENV PATH=/root/.local/bin:${PATH}

COPY requirements.txt ./
RUN pip install --user -r requirements.txt

COPY requirements.cpu.txt ./
RUN pip install --user -r requirements.cpu.txt

COPY requirements.delft.txt ./
RUN pip install --user -r requirements.delft.txt --no-deps

ARG install_dev
COPY requirements.dev.txt ./
RUN if [ "${install_dev}" = "y" ]; then pip install -r requirements.dev.txt; fi

COPY sciencebeam_trainer_delft ./sciencebeam_trainer_delft
COPY setup.py ./

COPY .flake8 .pylintrc pytest.ini ./
COPY tests ./tests
