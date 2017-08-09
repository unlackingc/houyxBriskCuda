
#include "../../includes/AgastCuda.h"


    __host__ __device__ Agast::Agast( int step )
	{


		 const int (*offsets)[2] = offsets8;

		    assert(pixel && offsets);

		    int k = 0;
		    for( ; k < 16; k++ )
		        pixel[k] = offsets[k][0] + offsets[k][1] * step;
	};

    __host__ __device__ Agast::Agast(  const Agast& c )
	{
		 *this = c;
	};

	__device__ int Agast::agast_cornerScore_5_8( const unsigned char* ptr, int threshold) const
	{
	    int bmin = threshold;
	    int bmax = 255;
	    int b_test = (bmax + bmin)/2;

	    short Offset58_0 = (short) pixel[0];
	    short Offset58_1 = (short) pixel[1];
	    short Offset58_2 = (short) pixel[2];
	    short Offset58_3 = (short) pixel[3];
	    short Offset58_4 = (short) pixel[4];
	    short Offset58_5 = (short) pixel[5];
	    short Offset58_6 = (short) pixel[6];
	    short Offset58_7 = (short) pixel[7];

	    while(true)
	    {
	        const int cb = *ptr + b_test;
	        const int c_b = *ptr - b_test;
	        if(ptr[Offset58_0] > cb)
	          if(ptr[Offset58_2] > cb)
	            if(ptr[Offset58_3] > cb)
	              if(ptr[Offset58_5] > cb)
	                if(ptr[Offset58_1] > cb)
	                  if(ptr[Offset58_4] > cb)
	                    goto is_a_corner;
	                  else
	                    if(ptr[Offset58_7] > cb)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  if(ptr[Offset58_4] > cb)
	                    if(ptr[Offset58_6] > cb)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	              else
	                if(ptr[Offset58_1] > cb)
	                  if(ptr[Offset58_4] > cb)
	                    goto is_a_corner;
	                  else
	                    if(ptr[Offset58_7] > cb)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  goto is_not_a_corner;
	            else
	              if(ptr[Offset58_7] > cb)
	                if(ptr[Offset58_6] > cb)
	                  if(ptr[Offset58_5] > cb)
	                    if(ptr[Offset58_1] > cb)
	                      goto is_a_corner;
	                    else
	                      if(ptr[Offset58_4] > cb)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                  else
	                    if(ptr[Offset58_1] > cb)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  goto is_not_a_corner;
	              else
	                if(ptr[Offset58_5] < c_b)
	                  if(ptr[Offset58_3] < c_b)
	                    if(ptr[Offset58_7] < c_b)
	                      if(ptr[Offset58_4] < c_b)
	                        if(ptr[Offset58_6] < c_b)
	                          goto is_a_corner;
	                        else
	                          goto is_not_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	                else
	                  goto is_not_a_corner;
	          else
	            if(ptr[Offset58_5] > cb)
	              if(ptr[Offset58_7] > cb)
	                if(ptr[Offset58_6] > cb)
	                  if(ptr[Offset58_1] > cb)
	                    goto is_a_corner;
	                  else
	                    if(ptr[Offset58_4] > cb)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  goto is_not_a_corner;
	              else
	                goto is_not_a_corner;
	            else
	              if(ptr[Offset58_5] < c_b)
	                if(ptr[Offset58_3] < c_b)
	                  if(ptr[Offset58_2] < c_b)
	                    if(ptr[Offset58_1] < c_b)
	                      if(ptr[Offset58_4] < c_b)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      if(ptr[Offset58_4] < c_b)
	                        if(ptr[Offset58_6] < c_b)
	                          goto is_a_corner;
	                        else
	                          goto is_not_a_corner;
	                      else
	                        goto is_not_a_corner;
	                  else
	                    if(ptr[Offset58_7] < c_b)
	                      if(ptr[Offset58_4] < c_b)
	                        if(ptr[Offset58_6] < c_b)
	                          goto is_a_corner;
	                        else
	                          goto is_not_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  goto is_not_a_corner;
	              else
	                goto is_not_a_corner;
	        else if(ptr[Offset58_0] < c_b)
	          if(ptr[Offset58_2] < c_b)
	            if(ptr[Offset58_7] > cb)
	              if(ptr[Offset58_3] < c_b)
	                if(ptr[Offset58_5] < c_b)
	                  if(ptr[Offset58_1] < c_b)
	                    if(ptr[Offset58_4] < c_b)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    if(ptr[Offset58_4] < c_b)
	                      if(ptr[Offset58_6] < c_b)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  if(ptr[Offset58_1] < c_b)
	                    if(ptr[Offset58_4] < c_b)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	              else
	                if(ptr[Offset58_5] > cb)
	                  if(ptr[Offset58_3] > cb)
	                    if(ptr[Offset58_4] > cb)
	                      if(ptr[Offset58_6] > cb)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	                else
	                  goto is_not_a_corner;
	            else
	              if(ptr[Offset58_7] < c_b)
	                if(ptr[Offset58_3] < c_b)
	                  if(ptr[Offset58_5] < c_b)
	                    if(ptr[Offset58_1] < c_b)
	                      goto is_a_corner;
	                    else
	                      if(ptr[Offset58_4] < c_b)
	                        if(ptr[Offset58_6] < c_b)
	                          goto is_a_corner;
	                        else
	                          goto is_not_a_corner;
	                      else
	                        goto is_not_a_corner;
	                  else
	                    if(ptr[Offset58_1] < c_b)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  if(ptr[Offset58_6] < c_b)
	                    if(ptr[Offset58_5] < c_b)
	                      if(ptr[Offset58_1] < c_b)
	                        goto is_a_corner;
	                      else
	                        if(ptr[Offset58_4] < c_b)
	                          goto is_a_corner;
	                        else
	                          goto is_not_a_corner;
	                    else
	                      if(ptr[Offset58_1] < c_b)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	              else
	                if(ptr[Offset58_3] < c_b)
	                  if(ptr[Offset58_5] < c_b)
	                    if(ptr[Offset58_1] < c_b)
	                      if(ptr[Offset58_4] < c_b)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      if(ptr[Offset58_4] < c_b)
	                        if(ptr[Offset58_6] < c_b)
	                          goto is_a_corner;
	                        else
	                          goto is_not_a_corner;
	                      else
	                        goto is_not_a_corner;
	                  else
	                    if(ptr[Offset58_1] < c_b)
	                      if(ptr[Offset58_4] < c_b)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  goto is_not_a_corner;
	          else
	            if(ptr[Offset58_5] > cb)
	              if(ptr[Offset58_3] > cb)
	                if(ptr[Offset58_2] > cb)
	                  if(ptr[Offset58_1] > cb)
	                    if(ptr[Offset58_4] > cb)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    if(ptr[Offset58_4] > cb)
	                      if(ptr[Offset58_6] > cb)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  if(ptr[Offset58_7] > cb)
	                    if(ptr[Offset58_4] > cb)
	                      if(ptr[Offset58_6] > cb)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	              else
	                goto is_not_a_corner;
	            else
	              if(ptr[Offset58_5] < c_b)
	                if(ptr[Offset58_7] < c_b)
	                  if(ptr[Offset58_6] < c_b)
	                    if(ptr[Offset58_1] < c_b)
	                      goto is_a_corner;
	                    else
	                      if(ptr[Offset58_4] < c_b)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	                else
	                  goto is_not_a_corner;
	              else
	                goto is_not_a_corner;
	        else
	          if(ptr[Offset58_3] > cb)
	            if(ptr[Offset58_5] > cb)
	              if(ptr[Offset58_2] > cb)
	                if(ptr[Offset58_1] > cb)
	                  if(ptr[Offset58_4] > cb)
	                    goto is_a_corner;
	                  else
	                    goto is_not_a_corner;
	                else
	                  if(ptr[Offset58_4] > cb)
	                    if(ptr[Offset58_6] > cb)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	              else
	                if(ptr[Offset58_7] > cb)
	                  if(ptr[Offset58_4] > cb)
	                    if(ptr[Offset58_6] > cb)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	                else
	                  goto is_not_a_corner;
	            else
	              goto is_not_a_corner;
	          else
	            if(ptr[Offset58_3] < c_b)
	              if(ptr[Offset58_5] < c_b)
	                if(ptr[Offset58_2] < c_b)
	                  if(ptr[Offset58_1] < c_b)
	                    if(ptr[Offset58_4] < c_b)
	                      goto is_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    if(ptr[Offset58_4] < c_b)
	                      if(ptr[Offset58_6] < c_b)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      goto is_not_a_corner;
	                else
	                  if(ptr[Offset58_7] < c_b)
	                    if(ptr[Offset58_4] < c_b)
	                      if(ptr[Offset58_6] < c_b)
	                        goto is_a_corner;
	                      else
	                        goto is_not_a_corner;
	                    else
	                      goto is_not_a_corner;
	                  else
	                    goto is_not_a_corner;
	              else
	                goto is_not_a_corner;
	            else
	              goto is_not_a_corner;

	        is_a_corner:
	            bmin=b_test;
	            goto end;

	        is_not_a_corner:
	            bmax=b_test;
	            goto end;

	        end:

	        if(bmin == bmax - 1 || bmin == bmax)
	            return bmin;
	        b_test = (bmin + bmax) / 2;
	    }
	}
