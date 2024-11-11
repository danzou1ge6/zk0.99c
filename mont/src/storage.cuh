#ifndef STORAGE_H
#define STORAGE_H


namespace mont {
  
  using u32 = u_int32_t;
  using u64 = u_int64_t;
  using i64 = int64_t;
  using i64 = int64_t;
  using usize = size_t;

  template <usize N>
  struct ContinuousStorage {
    u32 limbs[N];

    u32 &operator[](usize i) {
      return limbs[i];
    }
  };

  template <usize N, usize STRIDE>
  struct StridedStorage {
    u32* p;

    u32 &operator[](usize i) {
      return *(p + i * STRIDE);
    }
  };
}

#endif