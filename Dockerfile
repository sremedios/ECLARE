FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
ENV PYTHONUSERBASE=/opt/conda
RUN apt-get update && apt-get install -y --no-install-recommends git

COPY . /tmp/smore

RUN pip install /tmp/smore && rm -rf /tmp/smore

ENTRYPOINT ["run-smore"]
