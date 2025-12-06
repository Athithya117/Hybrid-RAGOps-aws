.PHONY: s3 delete-s3 tree clean lc push docker-login docker-build-backup docker-push-backup

push:
	git config --global http.postBuffer 524288000
	git config --global http.maxRequestBuffer 1000M
	git config --global http.lowSpeedLimit 0
	git config --global http.lowSpeedTime 999999
	git add .
	git commit -m "new"
	git push origin main --force

s3:
	python3 utils/s3_buckets.py --create
	aws s3 ls "s3://$$S3_BUCKET/" --recursive | head -n 100

delete-s3:
	python3 utils/s3_buckets.py --create
	aws s3 ls

lc:
	bash utils/lc.sh

tree:
	tree -a -I '.git|.venv|repos|__pycache__|venv|commands.sh|production-stack|raw_data|.venv2|archive|tmp.md|docs|models|tmp|raw|chunked'

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.log" ! -path "./.git/*" -delete
	find . -type f -name "*.pulumi-logs" ! -path "./.git/*" -delete
	clear

docker-login:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin

index-image:
	bash apps/index/build_and_push_image.sh


init-reranker:
	python3 infra/generators/gen_reranker.py --generate

setup-flux:
	python3 infra/scripts/setup_fluxcd.py --auto-push

inspect-flux:
	tail -f infra/manifests/flux-system/setup_fluxcd.log

flux-status:
	flux check && flux get kustomizations -n flux-system

pulumi-up:
	bash infra/pulumi_aws/pulumi_setup.sh --create || true
pulumi-destroy:
	bash infra/pulumi_aws/pulumi_setup.sh --delete || true