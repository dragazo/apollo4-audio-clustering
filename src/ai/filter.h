#ifndef A3EM_FILTER_H
#define A3EM_FILTER_H

#include <cstring>
#include <algorithm>

template<typename T, int N, int E>
class Filter {
private:
    T means[N][E];
    T weights[N];
    const T base_radius_sqr;
    const T max_weight;

    static T l2_norm_sqr(const T a[E], const T b[E]) noexcept {
        T res = 0;
        for (int i = 0; i < E; ++i) {
            T c = a[i] - b[i];
            res += c * c;
        }
        return res;
    }

public:
    const T (&inspect_means() const noexcept)[N][E] { return means; }
    const T (&inspect_weights() const noexcept)[N] { return weights; }

    Filter(T _base_radius, T _max_weight) noexcept : base_radius_sqr(_base_radius * _base_radius), max_weight(_max_weight) {
        reset();
    }

    void reset() & noexcept {
        memset(means, 0, sizeof(means));
        memset(weights, 0, sizeof(weights));
    }

    bool insert(const T mean[E]) & noexcept {
        T center[E];
        T center_weight = 1;
        memcpy(center, mean, sizeof(center));

        int p = N;
        for (int i = N; i-- > 0; ) {
            if (Filter::l2_norm_sqr(means[i], mean) <= weights[i] * base_radius_sqr) {
                for (int j = 0; j < E; ++j) center[j] += means[i][j] * weights[i];
                center_weight += weights[i];
            } else {
                --p;
                if (p != i) {
                    memcpy(&means[p], &means[i], sizeof(*means));
                    memcpy(&weights[p], &weights[i], sizeof(*weights));
                }
            }
        }

        if (p == 0) p = 1;
        memmove(&means[p - 1], &means[p], (N - p) * sizeof(*means));
        memmove(&weights[p - 1], &weights[p], (N - p) * sizeof(*weights));
        --p;
        memset(&means[0], 0, p * sizeof(*means));
        memset(&weights[0], 0, p * sizeof(*weights));
        for (int i = 0; i < E; ++i) means[N - 1][i] = center[i] / center_weight;
        weights[N - 1] = std::min(center_weight, max_weight);

        return center_weight == 1;
    }
};

#endif
