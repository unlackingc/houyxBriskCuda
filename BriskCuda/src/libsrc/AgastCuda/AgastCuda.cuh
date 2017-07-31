/*
 * AgastCuda.cuh
 *
 *  Created on: 2017年7月31日
 *      Author: houyx
 */

#ifndef AGASTCUDA_CUH_
#define AGASTCUDA_CUH_
#include <stdio.h>
#include <stdlib.h>
#include "assert.h"
#include <cuda.h>


class Agast
{

public:

	int pixel[16];
	int offsets8[8][2] =
	{
		{-1,  0}, {-1, -1}, {0, -1}, { 1, -1},
		{ 1,  0}, { 1,  1}, {0,  1}, {-1,  1}
	};

    __host__ __device__ Agast( int step )
	{


		 const int (*offsets)[2] = offsets8;

		    assert(pixel && offsets);

		    int k = 0;
		    for( ; k < 16; k++ )
		        pixel[k] = offsets[k][0] + offsets[k][1] * step;
	};

	__device__ int agast_cornerScore_5_8( const unsigned char* ptr, int threshold) const
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

};



#endif /* AGASTCUDA_CUH_ */
