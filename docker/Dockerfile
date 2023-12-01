# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

RUN apt-get --allow-releaseinfo-change update
RUN apt-get install -y build-essential 

RUN mkdir /tmp/cellmaps_vnn
COPY ./ /tmp/cellmaps_vnn/
RUN pip install /tmp/cellmaps_vnn

RUN rm -rf /tmp/cellmaps_vnn

ENTRYPOINT ["/opt/conda/bin/cellmaps_vnncmd.py"]

CMD ["--help"]
