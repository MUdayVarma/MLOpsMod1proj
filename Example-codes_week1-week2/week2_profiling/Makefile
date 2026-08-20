# Week 2 -- only the C cache demo needs compiling; Python demos just run.
# Uses the system C compiler (clang on macOS, gcc on Linux). No OpenMP needed.
CC     ?= cc
CFLAGS ?= -O2 -Wall

all: cache_locality

cache_locality: cache_locality.c
	$(CC) $(CFLAGS) cache_locality.c -o cache_locality

# Convenience: build and immediately demo the layout slowdown.
run: cache_locality
	./cache_locality

clean:
	rm -f cache_locality *.png
