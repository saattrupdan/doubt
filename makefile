release-major:
	pytest && \
	python bump_version.py --major && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New major release'

release-minor:
	pytest && \
	python bump_version.py --minor && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New minor release'

release-patch:
	pytest && \
	python bump_version.py --patch && \
	git add doubt/__init__.py && \
	git commit -m 'feat: New patch release'
