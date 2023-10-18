FROM python:3.10.12
RUN pip install notebook==7.0.6
RUN useradd -u 1000 docker_user
RUN mkdir /home/docker_user
RUN chown -R docker_user:docker_user /home/docker_user
USER docker_user

