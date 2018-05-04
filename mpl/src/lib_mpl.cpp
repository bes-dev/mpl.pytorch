#include <TH/TH.h>
#include <limits>
#include <cmath>

extern "C" void compute_weights(int size,
                                const THFloatTensor *losses,
                                const THLongTensor *indices,
                                THFloatTensor *weights,
                                float ratio, float p)
{
    // int size = losses->size[0];
    const float* losses_data = THFloatTensor_data(losses);
    const int64_t* indices_data = THLongTensor_data(indices);
    float* weights_data = THFloatTensor_data(weights);

    // find first nonzero element
    int pos = 0;
    while( losses_data[pos] < std::numeric_limits<float>::epsilon() )
    {
        ++pos;
    }

    // Algorithm #1
    int n = size - pos;
    int m = int(ratio * n);
    if (n <= 0 || m <= 0) return;
    float q = p / (p - 1.0);
    int c = m - n + 1;
    float a[2] = {0.0};
    int i = pos;
    float eta = 0.0;
    for(; i < n && eta < std::numeric_limits<float>::epsilon(); ++i) {
        float loss_q = pow(losses_data[i] / losses_data[size - 1], q);
        a[0] = a[1];
        a[1] += loss_q;
        c += 1;
        eta = c * loss_q - a[1];
    }

    // compute alpha
    float alpha;
    if (eta < std::numeric_limits<float>::epsilon())
    {
        c += 1;
        a[0] = a[1];
    }
    alpha = pow(a[0] / c, 1.0 / q) * losses_data[size - 1];

    // compute weights
    float tau = 1.0 / (pow(n, 1.0 / q) * pow(m, 1.0 / p));
    for (int k = i; k < n; ++k)
    {
        weights_data[indices_data[k]] = tau;
    }
    if (alpha > -std::numeric_limits<float>::epsilon())
    {
        for(int k = pos; k < i; ++k)
        {
            weights_data[indices_data[k]] = tau * pow(losses_data[k] / alpha, q - 1);
        }
    }
}
