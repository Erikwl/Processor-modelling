import invoke

# tasks.py
invoke.run(
    "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC foo.cc "
    "-o libfoo.so "
)
