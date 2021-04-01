include .env
export $(shell sed 's/=.*//' .env)

release-major:
	pytest && \
	python bump_version.py --major && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New major release' && \
	make release

release-minor:
	pytest && \
	python bump_version.py --minor && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New minor release' && \
	make release

release-patch:
	pytest && \
	python bump_version.py --patch && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New patch release' && \
	make release

release:
	git push --tags origin master && \
	python setup.py sdist bdist_wheel && \
	twine upload dist/*
