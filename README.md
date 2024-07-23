# PopcornCuda

## build & run

Do an out-of-source build to a directory called 'build' then make the project:

```
cmake -S . -B build
cd build
make
```

Run it:
```
$ ./PopcornCuda 
Time to generate on GPU:  0.1 ms 
Time to generate on CPU:  1.0 ms 
```