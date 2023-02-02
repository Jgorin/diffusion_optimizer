ROOT_DIR:=C:/Users/andre/diffusion_optimizer

build:
	docker build . -t diffusion_optimizer

run:
	echo ${ROOT_DIR}/root/
	docker run -it --rm -e DISPLAY=${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix -v ${ROOT_DIR}/main:/home/main/ -v ${ROOT_DIR}/diffusion_optimizer:/home/diffusion_optimizer --shm-size 50G diffusion_optimizer /bin/bash