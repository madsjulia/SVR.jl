CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp
SHVER = 2
OS = $(shell uname)

all: svm-train modsvm-train lib denselib trainlib

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-fopenmp -dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-fopenmp -shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)
denselib: densesvm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-fopenmp -dynamiclib -Wl,-install_name,denselibsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-fopenmp -shared -Wl,-soname,denselibsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} densesvm.o -o denselibsvm.so.$(SHVER)
trainlib: modsvm-train.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-fopenmp -dynamiclib -Wl,-install_name,trainlib.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-fopenmp -shared -Wl,-soname,trainlib.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} modsvm-train.o svm.o -o trainlib.so.$(SHVER)
svm-train: svm-train.c svm.o
	$(CXX) $(CFLAGS) svm-train.c svm.o -o svm-train -lm
modsvm-train: svm.cpp modsvm-train.c modsvm-train.h
	$(CXX) $(CFLAGS) -c modsvm-train.c -o modsvm-train.o
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
densesvm.o: densesvm.cpp densesvm.h
	$(CXX) $(CFLAGS) -c densesvm.cpp
clean:
	rm -f *~ svm.o densesvm.o modsvm-train.o svm-train svm-predict svm-scale modsvm-train libsvm.so.$(SHVER) denselibsvm.so.$(SHVER) trainlib.so.$(SHVER)
