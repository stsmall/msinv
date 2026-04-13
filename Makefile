.PHONY: test clean

test:
	cd tests && python test_standard_coalescent.py
	cd tests && python test_msinv.py
	cd tests && python test_ld.py
	cd tests && python test_treeseq.py
	cd tests && python test_stdpopsim.py
	@echo "All tests passed."

clean:
	rm -rf msinv/__pycache__ tests/__pycache__
