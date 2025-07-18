push:
	git pull
	git add .
	git commit -m "new"
	git push origin main

s3:
	python3 utils/create_s3.py
	rm -rf pulumi backups

	