#include <immintrin.h>

//dont use this file if you run on ARM lol

#if defined(_MSC_VER) 
#define VECTORCALL __vectorcall
#elif defined(__GNUC__) || defined(__clang__)
#define VECTORCALL __attribute__((fastcall))
#else
#define VECTOCALL
#endif

enum class Arch {
    NONE,
    SSE,
    AVX2,
    AVX512
};

template<typename T, Arch arch> 
struct Vec {
};

template<>
struct Vec<float, Arch::SSE> {
    static constexpr int size() {
        return 4;
    }
    explicit Vec(__m128 raw)
        :raw_(raw) {}
    explicit Vec()
        :raw_{ _mm_setzero_ps() }
    {
    }
    explicit Vec(float val)
        :raw_{ _mm_set1_ps(val) }
    {
    }

    float reduce_add()const {
        __m128 sum = _mm_hadd_ps(raw_, raw_);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    Vec VECTORCALL fmadd(Vec x, Vec y)const {
        return Vec(_mm_fmadd_ps(raw_, x.raw_, y.raw_));
    }
    Vec VECTORCALL fmsub(Vec x, Vec y)const {
        return Vec(_mm_fmsub_ps(raw_, x.raw_, y.raw_));
    }

    Vec lowerhalf()const {
        return *this;
    }
    Vec upperhalf()const {
        return Vec(_mm_movehl_ps(raw_,raw_));
    }

    void stream(float * mem) {
        _mm_stream_ps(mem, raw_);
    }
    static Vec loadu(float const* mem) {
        return Vec(_mm_loadu_ps(mem));
    }
    static Vec load(float const* mem) {
        return Vec(_mm_load_ps(mem));
    }
    void storeu(float* mem)const {
        _mm_storeu_ps(mem, raw_);
    }
    void store(float* mem)const {
        _mm_store_ps(mem, raw_);
    }

    int VECTORCALL testz(Vec x) const{
        return _mm_testz_ps(raw_,x.raw_);
    }

    friend Vec VECTORCALL operator^ (Vec x, Vec y) {
        Vec(_mm_xor_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator &(Vec y , Vec x) {
        return Vec(_mm_and_ps(y.raw_, x.raw_));
    }
    friend Vec VECTORCALL operator |(Vec y, Vec x) {
        return Vec(_mm_or_ps(y.raw_, x.raw_));
    }

    Vec VECTORCALL max(Vec y)const {
        return Vec(_mm_max_ps(raw_, y.raw_));
    }

    Vec VECTORCALL min(Vec y)const {
        return Vec(_mm_min_ps(raw_, y.raw_));
    }


    Vec sqrt()const {
        return Vec(_mm_sqrt_ps(raw_));
    }

    int movemask()const {
        return _mm_movemask_ps(raw_);
    }

    Vec& VECTORCALL operator+=(Vec x) {
        raw_ = (*this + x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator*=(Vec x) {
        raw_ = (*this * x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator/=(Vec x) {
        raw_ = (*this / x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator-=(Vec x) {
        raw_ = (*this - x).raw_;
        return *this;
    }

    friend Vec VECTORCALL operator + (Vec x, Vec y) {
        return Vec(_mm_add_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator - (Vec x, Vec y) {
        return Vec(_mm_sub_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator* (Vec x, Vec y) {
        return Vec(_mm_mul_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator / (Vec x, Vec y) {
        return Vec(_mm_div_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator > (Vec x, Vec y) {
        return Vec(_mm_cmpgt_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator < (Vec x, Vec y) {
        return Vec(_mm_cmplt_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator >= (Vec x, Vec y) {
        return Vec(_mm_cmpge_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator <= (Vec x, Vec y) {
        return Vec(_mm_cmple_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator != (Vec x, Vec y) {
        return Vec(_mm_cmpneq_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator==(Vec x, Vec y) {
        return Vec(_mm_cmpeq_ps(x.raw_, y.raw_));
    }
private:
    __m128 raw_;
};

template<>
struct Vec<double, Arch::SSE> {
    static constexpr int size() {
        return 2;
    }
    explicit Vec(__m128d raw)
        :raw_(raw) {}
    explicit Vec()
        :raw_{ _mm_setzero_pd() }
    {
    }
    explicit Vec(double val)
        :raw_{ _mm_set1_pd(val) }
    {
    }
    Vec VECTORCALL max(Vec y)const {
        return Vec(_mm_max_pd(raw_, y.raw_));
    }
    Vec VECTORCALL min(Vec y)const {
        return Vec(_mm_min_pd(raw_, y.raw_));
    }
    Vec lowerhalf()const {
        return *this;
    }
    Vec upperhalf()const {
        return Vec(_mm_unpackhi_pd(raw_,raw_));
    }

    Vec VECTORCALL fmadd(Vec x, Vec y)const {
        return Vec(_mm_fmadd_pd(raw_, x.raw_, y.raw_));
    }
    Vec VECTORCALL fmsub(Vec x, Vec y)const {
        return Vec(_mm_fmsub_pd(raw_, x.raw_, y.raw_));
    }

    double reduce_add()const {
        __m128d sum = _mm_hadd_pd(raw_,raw_);
        return _mm_cvtsd_f64(sum);
    }

    int VECTORCALL testz(Vec x) const {
        return _mm_testz_pd(raw_, x.raw_);
    }
    friend Vec VECTORCALL operator^ (Vec x, Vec y) {
        Vec(_mm_xor_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator &(Vec y, Vec x) {
        return Vec(_mm_and_pd(y.raw_, x.raw_));
    }
    friend Vec VECTORCALL operator |(Vec y, Vec x) {
        return Vec(_mm_or_pd(y.raw_, x.raw_));
    }
    Vec sqrt()const {
        return Vec(_mm_sqrt_pd(raw_));
    }

    void stream(double* mem) {
        _mm_stream_pd(mem, raw_);
    }
    static Vec loadu(double const* mem) {
        return Vec(_mm_loadu_pd(mem));
    }
    static Vec load(double const* mem) {
        return Vec(_mm_load_pd(mem));
    }
    void storeu(double* mem)const {
        _mm_storeu_pd(mem, raw_);
    }
    void store(double* mem)const {
        _mm_store_pd(mem, raw_);
    }

    Vec& VECTORCALL operator+=(Vec x) {
        raw_ = (*this + x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator*=(Vec x) {
        raw_ = (*this * x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator/=(Vec x) {
        raw_ = (*this / x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator-=(Vec x) {
        raw_ = (*this - x).raw_;
        return *this;
    }

    int movemask()const {
        return _mm_movemask_pd(raw_);
    }

    friend Vec VECTORCALL operator + (Vec x, Vec y) {
        return Vec(_mm_add_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator - (Vec x, Vec y) {
        return Vec(_mm_sub_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator* (Vec x, Vec y) {
        return Vec(_mm_mul_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator / (Vec x, Vec y) {
        return Vec(_mm_div_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator > (Vec x, Vec y) {
        return Vec(_mm_cmpgt_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator < (Vec x, Vec y) {
        return Vec(_mm_cmplt_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator >= (Vec x, Vec y) {
        return Vec(_mm_cmpge_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator <= (Vec x, Vec y) {
        return Vec(_mm_cmple_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator != (Vec x, Vec y) {
        return Vec(_mm_cmpneq_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator==(Vec x, Vec y) {
        return Vec(_mm_cmpeq_pd(x.raw_, y.raw_));
    }
private:
    __m128d raw_;
};

template<>
struct Vec<float, Arch::AVX2> {
    static constexpr int size() {
        return 8;
    }
    explicit Vec(__m256 raw)
        :raw_(raw) {}
    explicit Vec()
        :raw_{ _mm256_setzero_ps() }
    {
    }
    explicit Vec(float val)
        :raw_{ _mm256_set1_ps(val) }
    {
    }
    Vec VECTORCALL max(Vec y)const {
        return Vec(_mm256_max_ps(raw_, y.raw_));
    }

    Vec VECTORCALL min(Vec y)const {
        return Vec(_mm256_min_ps(raw_, y.raw_));
    }
    Vec VECTORCALL fmadd(Vec x, Vec y)const {
        return Vec(_mm256_fmadd_ps(raw_, x.raw_, y.raw_));
    }
    Vec VECTORCALL fmsub(Vec x, Vec y)const {
        return Vec(_mm256_fmsub_ps(raw_, x.raw_, y.raw_));
    }

    friend Vec VECTORCALL operator^ (Vec x, Vec y) {
        Vec(_mm256_xor_ps(x.raw_, y.raw_));
    }
    float reduce_add()const { 
        return (lowerhalf() + upperhalf()).reduce_add();
    }

    Vec<float, Arch::SSE> lowerhalf()const {
        return Vec<float,Arch::SSE>(_mm256_castps256_ps128(raw_));
    }
    Vec<float, Arch::SSE> upperhalf()const {
        return Vec<float, Arch::SSE>(_mm256_extractf128_ps(raw_,1));
    }

    void stream(float* mem) {
        _mm256_stream_ps(mem, raw_);
    }
    static Vec loadu(float const* mem) {
        return Vec(_mm256_loadu_ps(mem));
    }
    static Vec load(float const* mem) {
        return Vec(_mm256_load_ps(mem));
    }
    void storeu(float* mem)const {
        _mm256_storeu_ps(mem, raw_);
    }
    void store(float* mem)const {
        _mm256_store_ps(mem, raw_);
    }
    int VECTORCALL testz(Vec x) const {
        return _mm256_testz_ps(raw_, x.raw_);
    }
    friend Vec VECTORCALL operator &(Vec y, Vec x) {
        return Vec(_mm256_and_ps(y.raw_, x.raw_));
    }
    friend Vec VECTORCALL operator |(Vec y, Vec x) {
        return Vec(_mm256_or_ps(y.raw_, x.raw_));
    }
    Vec sqrt()const {
        return Vec(_mm256_sqrt_ps(raw_));
    }

    int movemask()const {
        return _mm256_movemask_ps(raw_);
    }

    Vec& VECTORCALL operator+=(Vec x) {
        raw_ = (*this + x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator*=(Vec x) {
        raw_ = (*this * x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator/=(Vec x) {
        raw_ = (*this / x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator-=(Vec x) {
        raw_ = (*this - x).raw_;
        return *this;
    }


    friend Vec VECTORCALL operator + (Vec x, Vec y) {
        return Vec(_mm256_add_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator - (Vec x, Vec y) {
        return Vec(_mm256_sub_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator* (Vec x, Vec y) {
        return Vec(_mm256_mul_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator / (Vec x, Vec y) {
        return Vec(_mm256_div_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator > (Vec x, Vec y) {
        return Vec(_mm256_cmp_ps(x.raw_, y.raw_,_CMP_GT_OS));
    }
    friend Vec VECTORCALL operator < (Vec x, Vec y) {
        return Vec(_mm256_cmp_ps(x.raw_, y.raw_,_CMP_LT_OS));
    }
    friend Vec VECTORCALL operator >= (Vec x, Vec y) {
        return Vec(_mm256_cmp_ps(x.raw_, y.raw_,_CMP_GE_OS));
    }
    friend Vec VECTORCALL operator <= (Vec x, Vec y) {
        return Vec(_mm256_cmp_ps(x.raw_, y.raw_, _CMP_LE_OS));
    }
    friend Vec VECTORCALL operator != (Vec x, Vec y) {
        return Vec(_mm256_cmp_ps(x.raw_, y.raw_, _CMP_EQ_OQ));
    }
    friend Vec VECTORCALL operator==(Vec x, Vec y) {
        return Vec(_mm256_cmp_ps(x.raw_, y.raw_, _CMP_NEQ_UQ));
    }
private:
    __m256 raw_;
};

template<>
struct Vec<double, Arch::AVX2> {
    static constexpr int size() {
        return 4;
    }
    explicit Vec(__m256d raw)
        :raw_(raw) {}
    explicit Vec()
        :raw_{ _mm256_setzero_pd() }
    {
    }
    explicit Vec(double val)
        :raw_{ _mm256_set1_pd(val) }
    {
    }
    Vec VECTORCALL max(Vec y)const {
        return Vec(_mm256_max_pd(raw_, y.raw_));
    }

    Vec VECTORCALL min(Vec y)const {
        return Vec(_mm256_min_pd(raw_, y.raw_));
    }
    Vec VECTORCALL fmadd(Vec x, Vec y)const {
        return Vec(_mm256_fmadd_pd(raw_, x.raw_, y.raw_));
    }
    Vec VECTORCALL fmsub(Vec x, Vec y)const {
        return Vec(_mm256_fmsub_pd(raw_, x.raw_, y.raw_));
    }

    friend Vec VECTORCALL operator^ (Vec x, Vec y) {
        Vec(_mm256_xor_pd(x.raw_, y.raw_));
    }
    double reduce_add()const {
        return (lowerhalf() + upperhalf()).reduce_add();
    }

    Vec<double, Arch::SSE> lowerhalf()const {
        return Vec<double, Arch::SSE>(_mm256_castpd256_pd128(raw_));
    }
    Vec<double, Arch::SSE> upperhalf()const {
        return Vec<double, Arch::SSE>(_mm256_extractf128_pd(raw_, 1));
    }


    int VECTORCALL testz(Vec x) const {
        return _mm256_testz_pd(raw_, x.raw_);
    }
    friend Vec VECTORCALL operator &(Vec y, Vec x) {
        return Vec(_mm256_and_pd(y.raw_, x.raw_));
    }
    friend Vec VECTORCALL operator |(Vec y, Vec x) {
        return Vec(_mm256_or_pd(y.raw_, x.raw_));
    }
    Vec sqrt()const {
        return Vec(_mm256_sqrt_pd(raw_));
    }

    void stream(double* mem) {
        _mm256_stream_pd(mem, raw_);
    }
    static Vec loadu(double const* mem) {
        return Vec(_mm256_loadu_pd(mem));
    }
    static Vec load(double const* mem) {
        return Vec(_mm256_load_pd(mem));
    }
    void storeu(double* mem)const {
        _mm256_storeu_pd(mem, raw_);
    }
    void store(double* mem)const {
        _mm256_store_pd(mem, raw_);
    }

    int movemask()const {
        return _mm256_movemask_pd(raw_);
    }

    Vec& VECTORCALL operator+=(Vec x) {
        raw_ = (*this + x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator*=(Vec x) {
        raw_ = (*this * x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator/=(Vec x) {
        raw_ = (*this / x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator-=(Vec x) {
        raw_ = (*this - x).raw_;
        return *this;
    }

    friend Vec VECTORCALL operator + (Vec x, Vec y) {
        return Vec(_mm256_add_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator - (Vec x, Vec y) {
        return Vec(_mm256_sub_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator* (Vec x, Vec y) {
        return Vec(_mm256_mul_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator / (Vec x, Vec y) {
        return Vec(_mm256_div_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator > (Vec x, Vec y) {
        return Vec(_mm256_cmp_pd(x.raw_, y.raw_, _CMP_GT_OS));
    }
    friend Vec VECTORCALL operator < (Vec x, Vec y) {
        return Vec(_mm256_cmp_pd(x.raw_, y.raw_, _CMP_LT_OS));
    }
    friend Vec VECTORCALL operator >= (Vec x, Vec y) {
        return Vec(_mm256_cmp_pd(x.raw_, y.raw_, _CMP_GE_OS));
    }
    friend Vec VECTORCALL operator <= (Vec x, Vec y) {
        return Vec(_mm256_cmp_pd(x.raw_, y.raw_, _CMP_LE_OS));
    }
    friend Vec VECTORCALL operator != (Vec x, Vec y) {
        return Vec(_mm256_cmp_pd(x.raw_, y.raw_, _CMP_EQ_OQ));
    }
    friend Vec VECTORCALL operator==(Vec x, Vec y) {
        return Vec(_mm256_cmp_pd(x.raw_, y.raw_, _CMP_NEQ_UQ));
    }
private:
    __m256d raw_;
};



template<>
struct Vec<float, Arch::AVX512> {
    static constexpr int size() {
        return 16;
    }
    explicit Vec(__m512 raw)
        :raw_(raw) {}
    explicit Vec()
        :raw_{ _mm512_setzero_ps() }
    {
    }

    explicit Vec(float val)
        :raw_{ _mm512_set1_ps(val) }
    {
    }
    Vec VECTORCALL fmadd(Vec x, Vec y)const {
        return Vec(_mm512_fmadd_ps(raw_, x.raw_, y.raw_));
    }
    Vec VECTORCALL fmsub(Vec x, Vec y)const {
        return Vec(_mm512_fmsub_ps(raw_, x.raw_, y.raw_));
    }
    Vec VECTORCALL max(Vec y)const {
        return Vec(_mm512_max_ps(raw_, y.raw_));
    }

    Vec VECTORCALL min(Vec y)const {
        return Vec(_mm512_min_ps(raw_, y.raw_));
    }
    float reduce_add()const { 
        return (lowerhalf() + upperhalf()).reduce_add(); 
    }

    friend Vec VECTORCALL operator^ (Vec x, Vec y) {
        Vec(_mm512_xor_ps(x.raw_, y.raw_));
    }
    Vec<float, Arch::AVX2> lowerhalf()const {
        return Vec<float, Arch::AVX2>(_mm512_castps512_ps256(raw_));
    }
    Vec<float, Arch::AVX2> upperhalf()const {
        return Vec<float, Arch::AVX2>(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(raw_), 1)));
    }

    void stream(float* mem) {
        _mm512_stream_ps(mem, raw_);
    }
    static Vec loadu(float const* mem) {
        return Vec(_mm512_loadu_ps(mem));
    }
    static Vec load(float const* mem) {
        return Vec(_mm512_load_ps(mem));
    }
    void storeu(float* mem)const {
        _mm512_storeu_ps(mem, raw_);
    }
    void store(float* mem)const {
        _mm512_store_ps(mem, raw_);
    }

    friend Vec VECTORCALL operator &(Vec y, Vec x) {
        return Vec(_mm512_and_ps(y.raw_, x.raw_));
    }
    friend Vec VECTORCALL operator |(Vec y, Vec x) {
        return Vec(_mm512_or_ps(y.raw_, x.raw_));
    }
    Vec sqrt()const {
        return Vec(_mm512_sqrt_ps(raw_));
    }
    

    Vec& VECTORCALL operator+=(Vec x) {
        raw_ = (*this + x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator*=(Vec x) {
        raw_ = (*this * x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator/=(Vec x) {
        raw_ = (*this / x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator-=(Vec x) {
        raw_ = (*this - x).raw_;
        return *this;
    }


    friend Vec VECTORCALL operator + (Vec x, Vec y) {
        return Vec(_mm512_add_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator - (Vec x, Vec y) {
        return Vec(_mm512_sub_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator* (Vec x, Vec y) {
        return Vec(_mm512_mul_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator / (Vec x, Vec y) {
        return Vec(_mm512_div_ps(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator > (Vec x, Vec y) {
        return Vec(_mm512_cmp_ps_mask(x.raw_, y.raw_, _CMP_GT_OS));
    }
    friend Vec VECTORCALL operator < (Vec x, Vec y) {
        return Vec(_mm512_cmp_ps_mask(x.raw_, y.raw_, _CMP_LT_OS));
    }
    friend Vec VECTORCALL operator >= (Vec x, Vec y) {
        return Vec(_mm512_cmp_ps_mask(x.raw_, y.raw_, _CMP_GE_OS));
    }
    friend Vec VECTORCALL operator <= (Vec x, Vec y) {
        return Vec(_mm512_cmp_ps_mask(x.raw_, y.raw_, _CMP_LE_OS));
    }
    friend Vec VECTORCALL operator != (Vec x, Vec y) {
        return Vec(_mm512_cmp_ps_mask(x.raw_, y.raw_, _CMP_EQ_OQ));
    }
    friend Vec VECTORCALL operator==(Vec x, Vec y) {
        return Vec(_mm512_cmp_ps_mask(x.raw_, y.raw_, _CMP_NEQ_UQ));
    }
private:
    __m512 raw_;
};


template<>
struct Vec<double, Arch::AVX512> {
    static constexpr int size() {
        return 8;
    }
    explicit Vec(__m512d raw)
        :raw_(raw) {}
    explicit Vec()
        :raw_{ _mm512_setzero_pd() }
    {
    }
    explicit Vec(double val)
        :raw_{ _mm512_set1_pd(val) }
    {
    }
    Vec VECTORCALL max(Vec y)const {
        return Vec(_mm512_max_pd(raw_, y.raw_));
    }

    Vec VECTORCALL min(Vec y)const {
        return Vec(_mm512_min_pd(raw_, y.raw_));
    }

    Vec VECTORCALL fmadd(Vec x, Vec y)const { 
        return Vec(_mm512_fmadd_pd(raw_, x.raw_, y.raw_)); 
    } 
    Vec VECTORCALL fmsub(Vec x, Vec y)const {
        return Vec(_mm512_fmsub_pd(raw_, x.raw_, y.raw_));
    }

    double reduce_add()const { 
        return (lowerhalf() + upperhalf()).reduce_add(); 
    }

    friend Vec VECTORCALL operator^ (Vec x, Vec y) {
        Vec(_mm512_xor_pd(x.raw_, y.raw_));
    }
    Vec<double, Arch::AVX2> lowerhalf()const {
        return Vec<double, Arch::AVX2>(_mm512_castpd512_pd256(raw_));
    }
    Vec<double, Arch::AVX2> upperhalf()const {
        return Vec<double, Arch::AVX2>(_mm512_extractf64x4_pd(raw_,1));
    }

    friend Vec VECTORCALL operator &(Vec y, Vec x) {
        return Vec(_mm512_and_pd(y.raw_, x.raw_));
    }
    friend Vec VECTORCALL operator |(Vec y, Vec x) {
        return Vec(_mm512_or_pd(y.raw_, x.raw_));
    }
    Vec sqrt()const {
        return Vec(_mm512_sqrt_pd(raw_));
    }
    

    void stream(double* mem) {
        _mm512_stream_pd(mem, raw_);
    }
    static Vec loadu(double const* mem) {
        return Vec(_mm512_loadu_pd(mem));
    }
    static Vec load(double const* mem) {
        return Vec(_mm512_load_pd(mem));
    }
    void storeu(double* mem)const {
        _mm512_storeu_pd(mem, raw_);
    }
    void store(double* mem)const {
        _mm512_store_pd(mem, raw_);
    }

    
    Vec& VECTORCALL operator+=(Vec x) {
        raw_ = (*this + x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator*=(Vec x) {
        raw_ = (*this * x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator/=(Vec x) {
        raw_ = (*this / x).raw_;
        return *this;
    }

    Vec& VECTORCALL operator-=(Vec x) {
        raw_ = (*this - x).raw_;
        return *this;
    }

    friend Vec VECTORCALL operator + (Vec x, Vec y) {
        return Vec(_mm512_add_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator - (Vec x, Vec y) {
        return Vec(_mm512_sub_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator* (Vec x, Vec y) {
        return Vec(_mm512_mul_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator / (Vec x, Vec y) {
        return Vec(_mm512_div_pd(x.raw_, y.raw_));
    }
    friend Vec VECTORCALL operator > (Vec x, Vec y) {
        return Vec(_mm512_cmp_pd_mask(x.raw_, y.raw_, _CMP_GT_OS));
    }
    friend Vec VECTORCALL operator < (Vec x, Vec y) {
        return Vec(_mm512_cmp_pd_mask(x.raw_, y.raw_, _CMP_LT_OS));
    }
    friend Vec VECTORCALL operator >= (Vec x, Vec y) {
        return Vec(_mm512_cmp_pd_mask(x.raw_, y.raw_, _CMP_GE_OS));
    }
    friend Vec VECTORCALL operator <= (Vec x, Vec y) {
        return Vec(_mm512_cmp_pd_mask(x.raw_, y.raw_, _CMP_LE_OS));
    }
    friend Vec VECTORCALL operator != (Vec x, Vec y) {
        return Vec(_mm512_cmp_pd_mask(x.raw_, y.raw_, _CMP_EQ_OQ));
    }
    friend Vec VECTORCALL operator==(Vec x, Vec y) {
        return Vec(_mm512_cmp_pd_mask(x.raw_, y.raw_, _CMP_NEQ_UQ));
    }
private:
    __m512d raw_;
};
