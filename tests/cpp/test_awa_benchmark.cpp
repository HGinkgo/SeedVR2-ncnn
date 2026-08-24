#include <cmath>
#include <cstdio>
#include <vector>

#include "awa/awa_benchmark.h"

int main()
{
    if (std::fabs(seedvr2_awa_median_ms({7.0, 1.0, 5.0}) - 5.0) > 1e-12)
        return 1;
    if (std::fabs(seedvr2_awa_median_ms({9.0, 1.0, 7.0, 3.0}) - 5.0) > 1e-12)
        return 1;

    SeedVR2AwaBenchmarkShape shape;
    if (!seedvr2_awa_parse_shape("1,45,80,20,128,5", shape))
        return 1;
    if (shape.source_t != 1 || shape.source_h != 45 || shape.source_w != 80 || shape.heads != 20 ||
        shape.head_dim != 128 || shape.text_tokens != 5)
        return 1;
    if (seedvr2_awa_parse_shape("1,45,80,20,0,5", shape))
        return 1;

    std::puts("seedvr2-awa-benchmark: ok");
    return 0;
}
