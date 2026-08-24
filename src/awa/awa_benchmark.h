#pragma once

#include <vector>

struct SeedVR2AwaBenchmarkShape
{
    int source_t = 0;
    int source_h = 0;
    int source_w = 0;
    int heads = 0;
    int head_dim = 0;
    int text_tokens = 0;
};

double seedvr2_awa_median_ms(std::vector<double> samples);
bool seedvr2_awa_parse_shape(const char* value, SeedVR2AwaBenchmarkShape& shape);
