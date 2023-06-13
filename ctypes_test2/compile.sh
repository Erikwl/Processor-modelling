#!/bin/bash
g++ -c -fPIC mva_c.cc -o mva_c.o
g++ -shared -Wl,-soname,create_c_vector.so -o mva_c.so mva_c.o

