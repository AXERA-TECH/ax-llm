#include "../src/utils/num2words_en.hpp"

int main()
{
    Num2Word_EN num2word_en;

    for (int i = 0; i < 100; i++)
    {
        std::cout << i << " -> " << num2word_en[i] << std::endl;
    }
    return 0;
}