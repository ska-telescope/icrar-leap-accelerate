
#include <icrar/leap-accelerate/config.h>
#include <cstdlib>

std::string get_test_data_dir()
{
    if(const char* env_p = std::getenv("TEST_DATA_DIR"))
    {
        return std::string(env_p);
    }
    else
    {
        return std::string(TEST_DATA_DIR);
    }
}