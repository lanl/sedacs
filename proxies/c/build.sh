gcc-mp-12 -c proxy_a_lib.c -I/opt/local/include
gcc-mp-12 proxy_a.c -o proxy_a proxy_a_lib.o -L/opt/local/lib -lopenblas
