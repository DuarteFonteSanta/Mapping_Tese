
docker build -t somatosensory-mapping .

docker run -p 8888:8888 -v ${PWD}/mapping_tese/notebooks:/app/mapping_tese/notebooks somatosensory-mapping /app/start-jupyter.sh
   

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop <container_name>

# Remove a container
docker rm <container_name>

# List images
docker images

# Remove an image
docker rmi <image_name>

# View container logs
docker logs <container_name>

# Execute command in running container
docker exec -it <container_name> /bin/bash

# Rebuild after changing Dockerfile
docker-compose build --no-cache