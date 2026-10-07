/**
 * Your solution. Should match the CPU output.
 */
__global__
void opposing_sort( element_t * data, std::size_t invert_at_pos, std::size_t num_elements )
{
    int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if( th_id > num_elements ) {return;}
    {

        // IMPLEMENT ME!

        // bool ascending = (th_id < invert_at_pos); // 1 = true, 0 = false
        // int stride_max = ascending ? invert_at_pos / 2 : (num_elements - invert_at_pos) / 2;
        // int local_th_id = ascending ? th_id : th_id - invert_at_pos; // if ascending, keep th_id, else subtract invert_at_pos

        for(int stride = 1; stride <= num_elements/2; stride <<= 1){
            int global_group_size = stride * 2;
            int group_number = th_id/global_group_size;
            int dir = (group_number % 2 == 0); //if 1 then ascending, 0 then descending
            // dir = ascending ? dir : !dir; // if ascending, keep dir, else invert it

            int stride_temp =  stride;
            while (stride_temp >= 1){
                int group_size = stride_temp * 2;
                int head = (th_id / group_size) * group_size; //integer division
                
                if ((th_id < head + (group_size/2)) && (th_id + stride_temp < head + group_size)){ // choose a better way to find valid index
                    if ((data[th_id] < data[th_id + stride_temp]) != dir){ //if it don't match dir, swap
                        int temp = data[th_id];
                        data[th_id] = data[th_id + stride_temp];
                        data[th_id + stride_temp] = temp;
                    }
                }

                stride_temp >>= 1;

                __syncthreads();

            }
        }
    }

    return;
}