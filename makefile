include .env
export $(shell sed 's/=.*//' .env)

release-major:
	pytest && \
	python bump_version.py --major && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New major release' && \
	git pull origin master & \
	git push --tags origin master && \
	python setup.py sdist bdist_wheel && \
	twine upload dist/*

release-minor:
	pytest && \
	python bump_version.py --minor && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New minor release' && \
	git pull origin master & \
	git push --tags origin master && \
	python setup.py sdist bdist_wheel && \
	twine upload dist/*

release-patch:
	pytest && \
	python bump_version.py --patch && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New patch release' && \
	git pull origin master & \
	git push --tags origin master && \
	python setup.py sdist bdist_wheel && \
	twine upload dist/*
