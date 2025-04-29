#ifndef A3EM_FILTER_H
#define A3EM_FILTER_H

#include <cstring>
#include <algorithm>

template<typename T, int E, int N>
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

        int p = 0;
        for (int i = 0; i < N; ++i) {
            if (Filter::l2_norm_sqr(means[i], mean) <= weights[i] * base_radius_sqr) {
                for (int j = 0; j < E; ++j) center[j] += means[i][j] * weights[i];
                center_weight += weights[i];
            } else {
                if (p != i) {
                    memcpy(&means[p], &means[i], sizeof(*means));
                    memcpy(&weights[p], &weights[i], sizeof(*weights));
                }
                ++p;
            }
        }
        if (p == N) {
            memmove(&means[0], &means[1], (N - 1) * sizeof(*means));
            memmove(&weights[0], &weights[1], (N - 1) * sizeof(*weights));
            --p;
        }
        for (int i = 0; i < E; ++i) means[p][i] = center[i] / center_weight;
        weights[p] = std::min(center_weight, max_weight);
        ++p;
        memset(&means[p], 0, (N - p) * sizeof(*means));
        memset(&weights[p], 0, (N - p) * sizeof(*weights));

        return center_weight == 1;
    }
};

#endif
