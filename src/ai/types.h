#ifndef A3EM_TYPES_H
#define A3EM_TYPES_H

#include "stdint.h"

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;

typedef float f32;
typedef double f64;

template<typename T>
struct Complex {
    T real;
    T imag;
};

typedef Complex<f32> c32;
typedef Complex<f64> c64;

template<typename T> struct complicate { typedef Complex<T> type; };
template<typename T> struct complicate<Complex<T>> { typedef Complex<T> type; };
template<typename T> using complicate_t = typename complicate<T>::type;

template<typename T> struct simplify { typedef T type; };
template<typename T> struct simplify<Complex<T>> { typedef T type; };
template<typename T> using simplify_t = typename simplify<T>::type;

template<typename T> T sqr_mag(const T &v) { return v * v; }
template<typename T> T sqr_mag(const Complex<T> &v) { return v.real * v.real + v.imag * v.imag; }

template<typename T> T conj(const T &v) { return v; }
template<typename T> Complex<T> conj(const Complex<T> &v) { return {v.real, -v.imag}; }

template<typename T> Complex<T> inline operator+(const Complex<T> &a, const Complex<T> &b) { return { a.real + b.real, a.imag + b.imag }; }
template<typename T> Complex<T> inline operator-(const Complex<T> &a, const Complex<T> &b) { return { a.real - b.real, a.imag - b.imag }; }

template<typename T> Complex<T> inline operator*(const Complex<T> &a, const Complex<T> &b) { return { a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real }; }
template<typename T> Complex<T> inline operator*(const Complex<T> &x, T s) { return { x.real * s, x.imag * s }; }
template<typename T> Complex<T> inline operator*(T s, const Complex<T> &x) { return { x.real * s, x.imag * s }; }

#endif
