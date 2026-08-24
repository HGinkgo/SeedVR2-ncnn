#include "awa_benchmark.h"

#include <algorithm>
#include <cstdlib>

double seedvr2_awa_median_ms(std::vector<double> samples)
{
    if (samples.empty())
        return 0.0;

    const size_t middle = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + middle, samples.end());
    const double upper = samples[middle];
    if (samples.size() % 2 != 0)
        return upper;

    std::nth_element(samples.begin(), samples.begin() + middle - 1, samples.begin() + middle);
    return (samples[middle - 1] + upper) * 0.5;
}

bool seedvr2_awa_parse_shape(const char* value, SeedVR2AwaBenchmarkShape& shape)
{
    if (!value || !value[0])
        return false;

    shape = SeedVR2AwaBenchmarkShape();
    int* fields[] = {&shape.source_t, &shape.source_h, &shape.source_w,
                     &shape.heads, &shape.head_dim, &shape.text_tokens};
    const char* cursor = value;
    for (int index = 0; index < 6; index++)
    {
        char* end = 0;
        const long parsed = std::strtol(cursor, &end, 10);
        if (end == cursor || parsed < 1 || parsed > 1000000)
            return false;
        *fields[index] = static_cast<int>(parsed);
        if (index < 5)
        {
            if (*end != ',')
                return false;
            cursor = end + 1;
        }
        else if (*end != '\0')
        {
            return false;
        }
    }
    return true;
}
