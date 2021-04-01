release-major:
	pytest && \
	python bump_version.py --major && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New major release' && \
	git push --tags && \
	python setup.py sdist bdist_wheel && \
	twine upload_dist/*

release-minor:
	pytest && \
	python bump_version.py --minor && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New minor release' && \
	git push --tags && \
	python setup.py sdist bdist_wheel && \
	twine upload_dist/*

release-patch:
	pytest && \
	python bump_version.py --patch && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New patch release' && \
	git push --tags && \
	python setup.py sdist bdist_wheel && \
	twine upload_dist/*
