#include <iostream>
#include <vector>

using namespace std;

class Foo{
    public:
        void bar(vector<int> a){
            for (auto i : a) {
                cout << i;
            }
            std::cout << "Hello " << a[0] << '\n' << std::endl;
        }
};

extern "C" {
    Foo* Foo_new(){ return new Foo(); }
    void Foo_bar(Foo* foo, vector<int> a){ foo->bar(a); }
}
