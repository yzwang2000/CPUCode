#include <iostream>
#include <immintrin.h>
using namespace std;
 
struct struct_Test1
{
	char c;
	int  i;
	double d;
};
 
struct alignas(4) struct_Test2
{
	char c;
	int  i;
	double d;
};
 
struct alignas(16) struct_Test3
{
	char c;
	int  i;
	double d;
};
 
struct alignas(32) struct_Test4
{
	char c;
	int  i;
	double d;
};
 
int main(int argc)
{
	struct_Test1 test1;
	struct_Test2 test2;
	struct_Test3 test3;
	struct_Test4 test4;
 
	cout<<"char size:"<<sizeof(char)<<endl;
	cout<<"int size:"<<sizeof(int)<<endl;
	cout<<"double size:"<<sizeof(double)<<endl;
 
	cout<<"test1 size:"<<sizeof(test1)<<endl;
	cout<<"test2 size:"<<sizeof(test2)<<endl;
	cout<<"test3 size:"<<sizeof(test3)<<endl;
	cout<<"test4 size:"<<sizeof(test4)<<endl;
 
	system("pause");

	return 0;
}