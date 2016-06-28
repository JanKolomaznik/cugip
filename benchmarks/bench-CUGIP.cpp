#include <cstdio>
#include <cstdlib>


int run(const char *dataset_path);

int main(int argc,char** argv)
{
  const char* dataset_path = argc==2 ? argv[1] : "./dataset";
  return run(dataset_path);
}
