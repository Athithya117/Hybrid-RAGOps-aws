push:
	git add .
	git commit -m "new"
	git push origin main

install:
	bash utils/bootstrap-amd.sh

login:
	bash utils/login.sh

