push:
	git pull
	git add .
	git commit -m "new"
	git push origin main

install:
	bash utils/bootstrap-amd.sh

login:
	bash utils/login.sh

s3:
	python3 utils/create_s3.py