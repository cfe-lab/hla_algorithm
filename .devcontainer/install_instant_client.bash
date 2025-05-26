#! /usr/bin/env bash

mkdir /tmp/vendor
cp vendor/instantclient*.zip /tmp/vendor/
unzip /tmp/vendor/instantclient*.zip -d /opt/oracle
ln -s /opt/oracle/instantclient_* /opt/oracle/instantclient
rm -rf /tmp/vendor
