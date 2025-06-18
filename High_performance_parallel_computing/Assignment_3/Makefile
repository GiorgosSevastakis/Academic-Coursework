
# Makefile for Task 1 and 2 – OpenACC Parallelization and Profiling (Profile both, but loop only final)

CXX := nvc++
OPT := -O3 -fast
CXXFLAGS := $(OPT) -std=c++14 -Wall -march=native -g
ACCFLAGS := -acc -gpu=cc70 -Minfo=acc

.PHONY: all clean profile timecsv

all: sw_parallel_oldv sw_parallel

sw_parallel_oldv: sw_parallel_oldv.cpp
	$(CXX) $(CXXFLAGS) $(ACCFLAGS) $< -o $@

sw_parallel: sw_parallel.cpp
	$(CXX) $(CXXFLAGS) $(ACCFLAGS) $< -o $@

profile: sw_parallel_oldv sw_parallel
	nv-nsight-cu-cli --clock-control none ./sw_parallel_oldv --out profile_parallel_oldv.txt
	nv-nsight-cu-cli --clock-control none ./sw_parallel --out profile_parallel.txt

# Run time measurements for various numgangs values
timecsv: sw_parallel
	echo "Ngangs,Time" > Strong.csv
	for gangs in 1 2 4 8 16 32 64 128 256 512 1024; do \
		echo "Running with $$gangs gangs..."; \
		output=$$(./sw_parallel --numgangs $$gangs 2>&1); \
		time=$$(echo "$$output" | grep "elapsed time" | awk '{print $$(NF-1)}'); \
		echo "$$gangs,$$time" >> Strong.csv; \
	done

clean:
	rm -f sw_parallel_oldv sw_parallel profile_parallel_oldv.txt profile_parallel.txt *.o *.mod Strong.csv
