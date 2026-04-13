.PHONY: build test clean

CC = gcc
CFLAGS = -O3 -shared -fPIC -lm

build:
	$(CC) $(CFLAGS) -o msinv/libmsinv.so msinv/csrc/libmsinv.c
	@echo "Built msinv/libmsinv.so"

test: build
	cd tests && python test_standard_coalescent.py
	cd tests && python test_msinv.py
	cd tests && python test_ld.py
	cd tests && python test_treeseq.py
	cd tests && python test_stdpopsim.py

clean:
	rm -f msinv/libmsinv.so msinv/*.pyc msinv/__pycache__/*
