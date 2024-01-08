#include <cublas.h>

#define CUBLAS_CHECK_ERR(ans) { cublas_check((ans), __FILE__, __LINE__); }

inline void cublas_check(int code, const char *file, int line, bool abort=true)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {    
        fprintf(stderr,"CUBLAS_CHECK_ERR: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }        
}
