/*
© 2024. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
*/

extern "C"
{
    void dm_diag(double *, double *, double, int, int, double);
}

extern "C"
{
    void dm_mlsp2(double *, double *, double *, int, int);
}

extern "C"
{
    void dm_dnnsp2(double *, double *, double *, float *, float *, float *, float *, float *, void *, void *, int, int, void *, cudaStream_t *);
}

extern "C"
{
    void dm_movingmusp2(double *, double *, int, double, void *);
}

extern "C"
{
    void dm_goldensp2(double *, double *, int, double, void *);
}

extern "C"
{
    void dm_dnnprt(double *, double *, double *, double *, int, int, void *);
}

extern "C"
{
    void dm_pscheby(double *, double *, int, int, double);
}

extern "C"
{
    void *dev_alloc(size_t);
}

extern "C"
{
    void *set_stream(void);
}

extern "C"
{
    void set_device(int);
}

extern "C"
{
    int get_device();
}

extern "C"
{
    void *host_alloc_pinned(size_t);
}

extern "C"
{
    void *dev_alloc_managed(size_t);
}

extern "C"
{
    void memcpyHtoH(void *, void *, size_t);
}

extern "C"
{
    void memcpyDtoH(void *, void *, size_t);
}

extern "C"
{
    void memcpyHtoD(void *, void *, size_t);
}

extern "C"
{
    void memcpyasyncDtoH(void *, void *, size_t);
}

extern "C"
{
    void memcpyasyncHtoD(void *, void *, size_t);
}

extern "C"
{
    void dev_free(void *);
}

extern "C"
{
    void *cublasInit();
}
