void compute_weights(int size,
                     const THFloatTensor *losses,
                     const THLongTensor *indices,
                     THFloatTensor *weights,
                     float ratio, float p);
