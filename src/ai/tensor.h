#ifndef A3EM_AI_TENSOR_H
#define A3EM_AI_TENSOR_H

#include <type_traits>
#include <stdexcept>

#include "./types.h"

template<typename T, u32 D, std::enable_if_t<(D > 0), int> = 0>
class Tensor {
private:

    u32 dims[D];
    T *data;
    void (*deleter)(T*);

public:

    Tensor() : dims{0}, data{nullptr}, deleter{nullptr} {};
    ~Tensor() {
        if (deleter && data) deleter(data);
        data = nullptr;
    }

    template<typename ...Args, std::enable_if_t<sizeof...(Args) == D, int> = 0>
    Tensor(T *_data, void (*_deleter)(T*), Args ..._dims) : dims{static_cast<u32>(_dims)...}, data{_data}, deleter{_deleter} {};

    Tensor(const Tensor &other) = delete;
    Tensor &operator=(const Tensor &other) = delete;

    Tensor(Tensor &&other) : dims{0}, data{nullptr}, deleter{nullptr} {
        *this = static_cast<Tensor&&>(other);
    }
    Tensor &operator=(Tensor &&other) {
        if (this != &other) {
            if (deleter && data) deleter(data);
            deleter = other.deleter;

            for (u32 i = 0; i < D; ++i) {
                dims[i] = other.dims[i];
                other.dims[i] = 0;
            }

            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    Tensor into_owned() && {
        Tensor res { static_cast<Tensor&&>(*this) };
        if (!res.deleter && res.data) {
            u32 s = res.size();
            T *new_data = new T[s];
            for (u32 i = 0; i < s; ++i) new_data[i] = res.data[i];
            res.data = new_data;
            res.deleter = [](auto *v) { delete[] v; };
        }
        return res;
    }

    template<u32 i, std::enable_if_t<(i < D), int> = 0>
    u32 dim() const {
        return dims[i];
    }

    u32 size() const {
        u32 res = 1;
        for (u32 i = 0; i < D; ++i) res *= dims[i];
        return res;
    }

    template<typename ...Args, std::enable_if_t<sizeof...(Args) == D, int> = 0>
    T &operator()(Args ...pos) {
        return const_cast<T&>(const_cast<const Tensor*>(this)->operator()(pos...));
    }

    template<typename ...Args, std::enable_if_t<sizeof...(Args) == D, int> = 0>
    const T &operator()(Args ..._pos) const {
        u32 pos[D] = {static_cast<u32>(_pos)...};
        u32 p = 0;
        u32 s = 1;
        for (u32 i = D; i-- > 0; ) {
            if (pos[i] >= dims[i]) throw std::runtime_error("index out of bounds");
            p += pos[i] * s;
            s *= dims[i];
        }
        return data[p];
    }

    Tensor &maximum(T v) & {
        for (u32 i = size(); i-- > 0; ) data[i] = std::max(data[i], v);
        return *this;
    }

    Tensor &minimum(T v) & {
        for (u32 i = size(); i-- > 0; ) data[i] = std::min(data[i], v);
        return *this;
    }

    T max() {
        if (size() <= 0) throw std::runtime_error("attempt to get max of empty tensor");
        T res = data[0];
        for (u32 i = size(); i-- > 0; ) res = std::max(res, data[i]);
        return res;
    }

    T min() {
        if (size() <= 0) throw std::runtime_error("attempt to get min of empty tensor");
        T res = data[0];
        for (u32 i = size(); i-- > 0; ) res = std::min(res, data[i]);
        return res;
    }

    T sum() {
        T res = (T)0;
        for (u32 i = size(); i-- > 0; ) res += data[i];
        return res;
    }

    T mean() {
        return sum() / size();
    }

    T var() {
        T m = mean();
        T s = (T)0;
        for (u32 i = size(); i-- > 0; ) {
            T v = data[i] - m;
            s += v * v;
        }
        return s / size();
    }

    T std() {
        return std::sqrt(var());
    }

    Tensor &operator/=(T v) & {
        for (u32 i = size(); i-- > 0; ) data[i] /= v;
        return *this;
    }
};

#endif
