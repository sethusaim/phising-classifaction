#!bin/bash

echo "Updating system packages"

sudo apt update

sudo apt-get update

sudo apt upgrade -y

echo "Updatedsystem packages"

echo "Installing AWS cli"

sudo apt install awscli -y

echo "Installed AWS cli"

echo "Installing Docker"

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

echo "Installed Docker"
