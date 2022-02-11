default:
	@echo "Please select a target"

.PHONY: publish
publish:
	mv docs docs2
	cp -r docs2/build/html docs
	git add docs
	git commit -m 'Updated documentation'
	git push --force
	git reset HEAD~1
	rm -r docs
	mv docs2 docs

