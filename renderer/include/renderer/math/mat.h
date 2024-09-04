#pragma once

#include "renderer/core.h"
#include "renderer/math/math.h"
#include "renderer/utils/ds.h"

namespace lab
{

namespace math
{

namespace 
{

template <int N>
LAB_CPU_GPU inline void init(double m[N][N], int i, int j) {}

template <int N, typename... Args>
LAB_CPU_GPU inline void init(double m[N][N], int i, int j, double v, Args... args) 
{
    m[i][j] = v;
    if (++j == N) 
    {
        ++i;
        j = 0;
    }
    init<N>(m, i, j, args...);
}

template <int N>
LAB_CPU_GPU inline void init_diag(double m[N][N], int i) {}

template <int N, typename... Args>
LAB_CPU_GPU inline void init_diag(double m[N][N], int i, double v, Args... args) 
{
    m[i][i] = v;
    init_diag<N>(m, i + 1, args...);
}

}

// Matrix Definition
template <int N>
class Matrix 
{
public:
    LAB_CPU_GPU
    static Matrix zeros() 
    {
        Matrix m;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m.m[i][j] = 0;
        return m;
    }

    LAB_CPU_GPU
    Matrix() 
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i][j] = (i == j) ? 1 : 0;
    }
    LAB_CPU_GPU
    Matrix(const double mat[N][N]) 
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                m[i][j] = mat[i][j];
    }
    LAB_CPU_GPU
    Matrix(renderer::span<const double> t);
    template <typename... Args>
    LAB_CPU_GPU Matrix(double v, Args... args) 
    {
        static_assert(1 + sizeof...(Args) == N * N, "Incorrect number of values provided to Matrix constructor");
        init<N>(m, 0, 0, v, args...);
    }
    template <typename... Args>
    LAB_CPU_GPU static Matrix Diag(double v, Args... args) 
    {
        static_assert(1 + sizeof...(Args) == N, "Incorrect number of values provided to Matrix::Diag");
        Matrix m;
        init_diag<N>(m.m, 0, v, args...);
        return m;
    }

    LAB_CPU_GPU
    Matrix operator+(const Matrix &m) const 
    {
        Matrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] += m.m[i][j];
        return r;
    }

    LAB_CPU_GPU
    Matrix operator*(double s) const 
    {
        Matrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] *= s;
        return r;
    }
    LAB_CPU_GPU
    Matrix operator/(double s) const 
    {
        LAB_CHECK_NE(s, 0);
        Matrix r = *this;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r.m[i][j] /= s;
        return r;
    }

    LAB_CPU_GPU
    bool operator==(const Matrix<N> &m2) const 
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (m[i][j] != m2.m[i][j])
                    return false;
        return true;
    }

    LAB_CPU_GPU
    bool operator!=(const Matrix<N> &m2) const 
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (m[i][j] != m2.m[i][j])
                    return true;
        return false;
    }

    LAB_CPU_GPU
    bool operator<(const Matrix<N> &m2) const 
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                if (m[i][j] < m2.m[i][j])
                    return true;
                if (m[i][j] > m2.m[i][j])
                    return false;
            }
        return false;
    }

    LAB_CPU_GPU
    bool is_identity() const;

    std::string to_string() const;

    LAB_CPU_GPU
    renderer::span<const double> operator[](int i) const { return m[i]; }
    LAB_CPU_GPU
    renderer::span<double> operator[](int i) { return renderer::span<double>(m[i]); }
private:
    double m[N][N];
};

// Matrix Inline Methods
template <int N>
inline bool Matrix<N>::is_identity() const 
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) 
        {
            if (i == j) {
                if (m[i][j] != 1)
                    return false;
            } else if (m[i][j] != 0)
                return false;
        }
    return true;
}

// Matrix Inline Functions
template <int N>
LAB_CPU_GPU inline Matrix<N> operator*(double s, const Matrix<N> &m) 
{
    return m * s;
}

template <typename Tresult, int N, typename T>
LAB_CPU_GPU inline Tresult mul(const Matrix<N> &m, const T &v) 
{
    Tresult result;
    for (int i = 0; i < N; ++i) 
    {
        result[i] = 0;
        for (int j = 0; j < N; ++j)
            result[i] += m[i][j] * v[j];
    }
    return result;
}

template <int N>
LAB_CPU_GPU double determinant(const Matrix<N> &m);

template <>
LAB_CPU_GPU inline double determinant(const Matrix<3> &m) 
{
    double minor12 = math::difference_of_products(m[1][1], m[2][2], m[1][2], m[2][1]);
    double minor02 = math::difference_of_products(m[1][0], m[2][2], m[1][2], m[2][0]);
    double minor01 = math::difference_of_products(m[1][0], m[2][1], m[1][1], m[2][0]);
    return math::fma(m[0][2], minor01,
               math::difference_of_products(m[0][0], minor12, m[0][1], minor02));
}

template <int N>
LAB_CPU_GPU inline Matrix<N> transpose(const Matrix<N> &m);
template <int N>
LAB_CPU_GPU std::optional<Matrix<N>> inverse(const Matrix<N> &);

template <int N>
LAB_CPU_GPU Matrix<N> invert_or_exit(const Matrix<N> &m) 
{
    std::optional<Matrix<N>> inv = inverse(m);
    LAB_CHECK(inv.has_value());
    return *inv;
}

template <int N>
LAB_CPU_GPU inline Matrix<N> transpose(const Matrix<N> &m) 
{
    Matrix<N> r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            r[i][j] = m[j][i];
    return r;
}

template <>
LAB_CPU_GPU inline std::optional<Matrix<3>> inverse(const Matrix<3> &m) 
{
    double det = determinant(m);
    if (det == 0)
        return {};
    double invDet = 1 / det;

    Matrix<3> r;

    r[0][0] = invDet * math::difference_of_products(m[1][1], m[2][2], m[1][2], m[2][1]);
    r[1][0] = invDet * math::difference_of_products(m[1][2], m[2][0], m[1][0], m[2][2]);
    r[2][0] = invDet * math::difference_of_products(m[1][0], m[2][1], m[1][1], m[2][0]);
    r[0][1] = invDet * math::difference_of_products(m[0][2], m[2][1], m[0][1], m[2][2]);
    r[1][1] = invDet * math::difference_of_products(m[0][0], m[2][2], m[0][2], m[2][0]);
    r[2][1] = invDet * math::difference_of_products(m[0][1], m[2][0], m[0][0], m[2][1]);
    r[0][2] = invDet * math::difference_of_products(m[0][1], m[1][2], m[0][2], m[1][1]);
    r[1][2] = invDet * math::difference_of_products(m[0][2], m[1][0], m[0][0], m[1][2]);
    r[2][2] = invDet * math::difference_of_products(m[0][0], m[1][1], m[0][1], m[1][0]);

    return r;
}

template <int N, typename T>
LAB_CPU_GPU inline T operator*(const Matrix<N> &m, const T &v) 
{
    return mul<T>(m, v);
}

template <>
LAB_CPU_GPU inline Matrix<4> operator*(const Matrix<4> &m1, const Matrix<4> &m2) 
{
    Matrix<4> r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r[i][j] = math::inner_product(m1[i][0], m2[0][j], m1[i][1], m2[1][j], m1[i][2],
                                   m2[2][j], m1[i][3], m2[3][j]);
    return r;
}

template <>
LAB_CPU_GPU inline Matrix<3> operator*(const Matrix<3> &m1, const Matrix<3> &m2) 
{
    Matrix<3> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i][j] =
                math::inner_product(m1[i][0], m2[0][j], m1[i][1], m2[1][j], m1[i][2], m2[2][j]);
    return r;
}

template <int N>
LAB_CPU_GPU inline Matrix<N> operator*(const Matrix<N> &m1, const Matrix<N> &m2) 
{
    Matrix<N> r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            r[i][j] = 0;
            for (int k = 0; k < N; ++k)
                r[i][j] = math::fma(m1[i][k], m2[k][j], r[i][j]);
        }
    return r;
}

template <int N>
LAB_CPU_GPU inline Matrix<N>::Matrix(renderer::span<const double> t) 
{
    LAB_CHECK_EQ(N * N, t.size());
    for (int i = 0; i < N * N; ++i)
        m[i / N][i % N] = t[i];
}

template <int N>
LAB_CPU_GPU Matrix<N> operator*(const Matrix<N> &m1, const Matrix<N> &m2);

template <>
LAB_CPU_GPU inline double determinant(const Matrix<1> &m) 
{
    return m[0][0];
}

template <>
LAB_CPU_GPU inline double determinant(const Matrix<2> &m) 
{
    return math::difference_of_products(m[0][0], m[1][1], m[0][1], m[1][0]);
}

template <>
LAB_CPU_GPU inline double determinant(const Matrix<4> &m) 
{
    double s0 = math::difference_of_products(m[0][0], m[1][1], m[1][0], m[0][1]);
    double s1 = math::difference_of_products(m[0][0], m[1][2], m[1][0], m[0][2]);
    double s2 = math::difference_of_products(m[0][0], m[1][3], m[1][0], m[0][3]);

    double s3 = math::difference_of_products(m[0][1], m[1][2], m[1][1], m[0][2]);
    double s4 = math::difference_of_products(m[0][1], m[1][3], m[1][1], m[0][3]);
    double s5 = math::difference_of_products(m[0][2], m[1][3], m[1][2], m[0][3]);

    double c0 = math::difference_of_products(m[2][0], m[3][1], m[3][0], m[2][1]);
    double c1 = math::difference_of_products(m[2][0], m[3][2], m[3][0], m[2][2]);
    double c2 = math::difference_of_products(m[2][0], m[3][3], m[3][0], m[2][3]);

    double c3 = math::difference_of_products(m[2][1], m[3][2], m[3][1], m[2][2]);
    double c4 = math::difference_of_products(m[2][1], m[3][3], m[3][1], m[2][3]);
    double c5 = math::difference_of_products(m[2][2], m[3][3], m[3][2], m[2][3]);

    return (math::difference_of_products(s0, c5, s1, c4) + math::difference_of_products(s2, c3, -s3, c2) +
            math::difference_of_products(s5, c0, s4, c1));
}

template <int N>
LAB_CPU_GPU inline double determinant(const Matrix<N> &m) 
{
    Matrix<N - 1> sub;
    double det = 0;
    // Inefficient, but we don't currently use N>4 anyway..
    for (int i = 0; i < N; ++i) {
        // Sub-matrix without row 0 and column i
        for (int j = 0; j < N - 1; ++j)
            for (int k = 0; k < N - 1; ++k)
                sub[j][k] = m[j + 1][k < i ? k : k + 1];

        double sign = (i & 1) ? -1 : 1;
        det += sign * m[0][i] * determinant(sub);
    }
    return det;
}

template <>
LAB_CPU_GPU inline std::optional<Matrix<4>> inverse(const Matrix<4> &m) 
{
    double s0 = math::difference_of_products(m[0][0], m[1][1], m[1][0], m[0][1]);
    double s1 = math::difference_of_products(m[0][0], m[1][2], m[1][0], m[0][2]);
    double s2 = math::difference_of_products(m[0][0], m[1][3], m[1][0], m[0][3]);

    double s3 = math::difference_of_products(m[0][1], m[1][2], m[1][1], m[0][2]);
    double s4 = math::difference_of_products(m[0][1], m[1][3], m[1][1], m[0][3]);
    double s5 = math::difference_of_products(m[0][2], m[1][3], m[1][2], m[0][3]);

    double c0 = math::difference_of_products(m[2][0], m[3][1], m[3][0], m[2][1]);
    double c1 = math::difference_of_products(m[2][0], m[3][2], m[3][0], m[2][2]);
    double c2 = math::difference_of_products(m[2][0], m[3][3], m[3][0], m[2][3]);

    double c3 = math::difference_of_products(m[2][1], m[3][2], m[3][1], m[2][2]);
    double c4 = math::difference_of_products(m[2][1], m[3][3], m[3][1], m[2][3]);
    double c5 = math::difference_of_products(m[2][2], m[3][3], m[3][2], m[2][3]);

    double det = math::inner_product(s0, c5, -s1, c4, s2, c3, s3, c2, s5, c0, -s4, c1);
    if (det == 0)
        return {};
    double s = 1 / det;

    double inv[4][4] = {{s * math::inner_product(m[1][1], c5, m[1][3], c3, -m[1][2], c4),
                        s * math::inner_product(-m[0][1], c5, m[0][2], c4, -m[0][3], c3),
                        s * math::inner_product(m[3][1], s5, m[3][3], s3, -m[3][2], s4),
                        s * math::inner_product(-m[2][1], s5, m[2][2], s4, -m[2][3], s3)},

                       {s * math::inner_product(-m[1][0], c5, m[1][2], c2, -m[1][3], c1),
                        s * math::inner_product(m[0][0], c5, m[0][3], c1, -m[0][2], c2),
                        s * math::inner_product(-m[3][0], s5, m[3][2], s2, -m[3][3], s1),
                        s * math::inner_product(m[2][0], s5, m[2][3], s1, -m[2][2], s2)},

                       {s * math::inner_product(m[1][0], c4, m[1][3], c0, -m[1][1], c2),
                        s * math::inner_product(-m[0][0], c4, m[0][1], c2, -m[0][3], c0),
                        s * math::inner_product(m[3][0], s4, m[3][3], s0, -m[3][1], s2),
                        s * math::inner_product(-m[2][0], s4, m[2][1], s2, -m[2][3], s0)},

                       {s * math::inner_product(-m[1][0], c3, m[1][1], c1, -m[1][2], c0),
                        s * math::inner_product(m[0][0], c3, m[0][2], c0, -m[0][1], c1),
                        s * math::inner_product(-m[3][0], s3, m[3][1], s1, -m[3][2], s0),
                        s * math::inner_product(m[2][0], s3, m[2][2], s0, -m[2][1], s1)}};

    return Matrix<4>(inv);
}

extern template class Matrix<2>;
extern template class Matrix<3>;
extern template class Matrix<4>;

using Matrix2 = Matrix<2>;
using Matrix3 = Matrix<3>;
using Matrix4 = Matrix<4>;


}

}