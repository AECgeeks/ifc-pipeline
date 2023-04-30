#!/usr/bin/env bash

while sleep 1; do
        find /tmp -mindepth 1 -atime +2 -exec rm -v -rf {} +
done
