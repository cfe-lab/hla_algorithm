# Base image for PyEasyHLA development.

# This image is the basis of the project's devcontainer as well.

ARG PYTHON_VERSION="3.13.3-bookworm"

FROM python:${PYTHON_VERSION} AS base

RUN apt update -y && apt upgrade -y

# Install the vendored software.
ARG INSTANTCLIENT_BASIC="instantclient-basic-linux.x64-23.7.0.25.01.zip"
# The value of ORACLE_HOME depends on the instant client used as it will
# install to a path like ".../instantclient_23_7" where the version numbers
# will vary.
ARG ORACLE_HOME="/opt/oracle/instantclient_23_7"

ENV ORACLE_HOME=${ORACLE_HOME} \
    LD_LIBRARY_PATH=${ORACLE_HOME}:$LD_LIBRARY_PATH

COPY vendor/${INSTANTCLIENT_BASIC} /tmp/vendor/
RUN unzip /tmp/vendor/${INSTANTCLIENT_BASIC} -d /opt/oracle &&\
    rm -rf /tmp/vendor
