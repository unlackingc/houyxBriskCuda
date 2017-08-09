#include "../includes/cuda_types.h"


void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
    if (err == cudaSuccess)
        return;
    std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
    exit (1);
}

template<typename T> __host__  bool cleanArray( T * ptr, int size,cudaStream_t& stream_ )
{

    CUDA_CHECK_RETURN(cudaMemsetAsync ( ptr, 0, sizeof(T)*size,stream_ ));

    return true;
}

template<typename T> __CV_CUDA_HOST_DEVICE__  T * newArrayIn( T * ptr, int size, bool ifset )
{
    ptr = new T[size];

    if( ifset )
    {
        memset(ptr, 0, sizeof(T)*size);
    }
    return ptr;
}

template<typename T> void pouta( T* m, int size,std::string info )
{
    std::cout << info << std::endl;
    for( int i = 0; i < size; i++  )
    {
        std::cout << i << " : ";
        pout( &m[i], info );
    }
    std::cout << "******************finish*******************" << std::endl;
}

template<typename T> void poutp(const PtrStepSz<T>& m, std::string info )
{
    std::cout << info << std::endl;
    for( int i = 0; i < m.rows; i++  )
    {
        for(int j = 0; j < m.cols; j ++)
        {
            std::cout <<"( " << i << " , " << j <<  " )";
            pout(m.at(i,j),info);
        }
    }
    std::cout << "******************finish*******************" << std::endl;
}

template<typename T> void pout( const T* ptr, std::string info,bool endline  )
{
    T temp;
    memset(&temp,0,sizeof(T));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(&temp, ptr, sizeof(T), cudaMemcpyDeviceToHost));
    std::cout << "===="<< info <<"====(" << temp << ")";
    std::cout << std::endl;
}
