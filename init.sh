#!/bin/bash

dotcount=`awk -F. '{print NF-1}' <<< "$1"`
if [ $dotcount -eq 1 ]; then
  domain="-d $1 -d www.$1"
elif [ $dotcount -eq 2 ]; then
  domain="-d $1"
else
  echo "usage: init.sh <domain>"
  exit 1
fi

sudo docker run -it --rm \
-v $PWD/docker-volumes/certbot/conf:/etc/letsencrypt \
-p 80:80 \
certbot/certbot \
certonly --standalone \
--register-unsafely-without-email --agree-tos \
--cert-name host \
$domain

# sudo openssl dhparam -out $PWD/certbot/dh-param/dhparam-2048.pem 2048
