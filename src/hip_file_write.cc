/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
** Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "mpi.h"

#include <hip/hip_runtime.h>
#include <chrono>

#include "hip_mpitest_utils.h"
#include "hip_mpitest_buffer.h"

int elements=64*1024*1024;
hip_mpitest_buffer *sendbuf=NULL;
hip_mpitest_buffer *recvbuf=NULL;

static void SL_read ( int hdl, void *buf, size_t num);


static void init_sendbuf (long *sendbuf, int count, int unused)
{
    for (long i = 0; i < count; i++) {
        sendbuf[i] = i+1;
    }
}

static void init_recvbuf (long *recvbuf, int count)
{
    for (long i = 0; i < count; i++) {
        recvbuf[i] = 0.0;
    }
}

static bool check_recvbuf(long *recvbuf, int nprocs_unused, int rank_unused, int count)
{
    bool res=true;

    for (long i=0; i<count; i++) {
        if (recvbuf[i] != i+1) {
            res = false;
#ifdef VERBOSE
            printf("recvbuf[%d] = %ld\n", i, recvbuf[i]);
#endif
        }
    }

    return res;
}

int file_write_test (void *sendbuf, int count,
                     MPI_Datatype datatype, MPI_File fh);

int main (int argc, char *argv[])
{
    int res;
    int rank, size;
    int root = 0;
    MPI_File fh;
    
    bind_device();

    MPI_Init      (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    parse_args(argc, argv, MPI_COMM_WORLD);

    long *tmp_sendbuf=NULL, *tmp_recvbuf=NULL;

    // Initialise send buffer
    ALLOCATE_SENDBUFFER(sendbuf, tmp_sendbuf, long, elements, sizeof(long),
                        rank, MPI_COMM_WORLD, init_sendbuf);

    // Initialize recv buffer
    ALLOCATE_RECVBUFFER(recvbuf, tmp_recvbuf, long, elements, sizeof(long),
                        rank, MPI_COMM_WORLD, init_recvbuf);

    // execute file_write test
    MPI_File_open(MPI_COMM_SELF, "testout.out", MPI_MODE_CREATE|MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1s = std::chrono::high_resolution_clock::now();
    res = file_write_test (sendbuf->get_buffer(), elements,
                           MPI_LONG, fh);
    if (MPI_SUCCESS != res) {
        fprintf(stderr, "Error in file_write_test. Aborting\n");
        MPI_Abort (MPI_COMM_WORLD, 1);
        return 1;
    }
    auto t1e = std::chrono::high_resolution_clock::now();
    double t1 = std::chrono::duration<double>(t1e-t1s).count();
    MPI_File_close (&fh);
    
    // verify results 
    bool ret = true;
    int fd = open ("testout.out", O_RDONLY );
    if ( -1 != fd ) {
        SL_read(fd, tmp_recvbuf, elements * sizeof(long));
        ret = check_recvbuf(tmp_recvbuf, size, rank, elements);
    }

    bool fret = report_testresult(argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(),
                                  recvbuf->get_memchar(), ret);
    report_performance (argv[0], MPI_COMM_WORLD, sendbuf->get_memchar(), recvbuf->get_memchar(),
                        elements, (size_t)(elements * sizeof(long)), 1, t1);

    //Free buffers
    FREE_BUFFER(sendbuf, tmp_sendbuf);
    FREE_BUFFER(recvbuf, tmp_recvbuf);

    delete (sendbuf);
    delete (recvbuf);

    MPI_Finalize ();    
    return fret ? 0 : 1;
}

int file_write_test (void *sendbuf, int count, MPI_Datatype datatype, MPI_File fh )
{
    int ret;

#ifdef HIP_MPITEST_FILE_IWRITE
    MPI_Request req;
    ret = MPI_File_iwrite(fh, sendbuf, count, datatype, &req);
    MPI_Wait (&req, MPI_STATUS_IGNORE);
#else
    ret = MPI_File_write (fh, sendbuf, count, datatype, MPI_STATUS_IGNORE);
#endif
    return ret;
}

void SL_read ( int hdl, void *buf, size_t num )
{
    int lcount=0;
    int a;
    char *wbuf = (char *)buf;
    do {
        a = read (hdl, wbuf, num);

        if(0 == a && num > 0){
            printf("\nSL_read: Warning: # Bytes read are less than "
                   "expected file size %d %s\n", hdl, strerror(errno));
            return;
        }

        if ( a == -1 ) {
            if (errno == EINTR       || errno == EAGAIN     ||
                errno == EINPROGRESS || errno == EWOULDBLOCK) {
                continue;
            } else {
                printf("SL_read: error while reading from file %d %s\n", hdl, strerror(errno));
                return;
            }
            lcount++;
            a=0;
        }

        num -= a;
        wbuf += a;
        
    } while ( num > 0 &&  lcount < 20 );

    return;
}
