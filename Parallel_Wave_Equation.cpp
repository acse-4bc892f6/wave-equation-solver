#include <mpi.h>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include <vector>
#include <fstream>
#include <sstream>

#define DO_TIMING

// total number of processes p, each with unique number id
int id, p; 

// number identifying the peer-to-peer communication
int tag_num = 1; 

/* discretisation and time stepping variables */
/* assume i is in row direction and j is in column direction */

// number of points in i and j direction in the whole domain
int imax = 500, jmax = 500;

// domain size in terms of x and y
// dx and dy calculated with jmax and imax
double y_max = 10.0, x_max = 10.0, dx, dy;

// maximum time step when iteration stops
double t_max = 10.0;

// grid data is written to file every dt_out seconds
// t_out is incremented by dt_out after writing file
// t is incremented by dt after each iteration
double t, t_out = 0.0, dt_out = 0.04, dt;

// boundary condition - Dirichlet (d) or Neumann (n)
char bc = 'n';

// speed of wave propagation
double c = 1;


class Wave
{
public:

    // start and end points of the decomposed domain
    int i_start, i_end, j_start, j_end; // in terms of i and j
    int y_start, y_end, x_start, x_end; // in terms of x and y

    // number of points in i and j direction
    int block_i, block_j; 

    // number of blocks divided in i and j direction based on number of processes
    int i_num, j_num;

    // 2d grids
    double **grid, **new_grid, **old_grid;
    // 1d grids
    double *grid_1d, *new_grid_1d, *old_grid_1d;

    // vector storing ids of neighbour blocks
    std::vector<int> neighbour_ids;

    // vector storing ids of stencil blocks
    std::vector<int> stencil_ids;

    void buildMPITypes();

    MPI_Datatype send_left, send_right, recv_left, recv_right;
    MPI_Datatype send_upper, send_lower, recv_upper, recv_lower;

    void setup_grids();
    void free_grids();
    void domain_decomposition();
    void find_neighbours();
    void initial_disturbance(double r_splash, double x_splash, double y_splash);
    void grid_to_file();
    void parameters_to_file();
    void domain_to_file();
    void get_ghost_layer();
    void do_iteration();
};


// build mpi datatypes to send and receive appropriate data between stencils without the need of copying data
void Wave::buildMPITypes()
{
    std::vector<int> block_lengths;
	std::vector<MPI_Aint> displacements;
	MPI_Aint add_start;
	std::vector<MPI_Datatype> typelist;

    // send to block to the left of current block - send first column
    MPI_Get_address(&grid[1][1], &add_start);

    block_lengths.resize(block_i);
    displacements.resize(block_i);
    typelist.resize(block_i);

    for (int i = 0; i < block_i; i++)
    {
        typelist[i] = MPI_DOUBLE;
        block_lengths[i] = 1;
        MPI_Get_address(&grid[i+1][1], &displacements[i]);
        displacements[i] = displacements[i] - add_start;
    }

    MPI_Type_create_struct(block_i, block_lengths.data(), displacements.data(), typelist.data(), &send_left);
    MPI_Type_commit(&send_left);

    // send to block to the right of current block - send last column

    MPI_Get_address(&grid[1][block_j], &add_start);

    for (int i = 0; i < block_i; i++)
    {
        typelist[i] = MPI_DOUBLE;
        block_lengths[i] = 1;
        MPI_Get_address(&grid[i+1][block_j], &displacements[i]);
        displacements[i] = displacements[i] - add_start;
    }

    MPI_Type_create_struct(block_i, block_lengths.data(), displacements.data(), typelist.data(), &send_right);
    MPI_Type_commit(&send_right);

    // receive from block to the left of current block - allocate to left ghost layer

    MPI_Get_address(&grid[1][0], &add_start);

    for (int i = 0; i < block_i; i++)
    {
        typelist[i] = MPI_DOUBLE;
        block_lengths[i] = 1;
        MPI_Get_address(&grid[i+1][0], &displacements[i]);
        displacements[i] = displacements[i] - add_start;
    }

    MPI_Type_create_struct(block_i, block_lengths.data(), displacements.data(), typelist.data(), &recv_left);
    MPI_Type_commit(&recv_left);

    // receive from block to the right of current block - allocate to right ghost layer

    MPI_Get_address(&grid[1][block_j+1], &add_start);

    for (int i = 0; i < block_i; i++)
    {
        typelist[i] = MPI_DOUBLE;
        block_lengths[i] = 1;
        MPI_Get_address(&grid[i+1][block_j+1], &displacements[i]);
        displacements[i] = displacements[i] - add_start;
    }

    MPI_Type_create_struct(block_i, block_lengths.data(), displacements.data(), typelist.data(), &recv_right);
    MPI_Type_commit(&recv_right);


    block_lengths.resize(block_j);
    displacements.resize(block_j);
    typelist.resize(block_j);

    // send to the block above current block - send top row

    MPI_Get_address(&grid[block_i][1], &add_start);

    for (int j = 0; j < block_j; j++)
    {
        typelist[j] = MPI_DOUBLE;
        block_lengths[j] = 1;
        MPI_Get_address(&grid[block_i][j+1], &displacements[j]);
        displacements[j] = displacements[j] - add_start;
    }

    MPI_Type_create_struct(block_j, block_lengths.data(), displacements.data(), typelist.data(), &send_upper);
    MPI_Type_commit(&send_upper);

    // send to the block below current block - send bottom row

    MPI_Get_address(&grid[1][1], &add_start);

    for (int j = 0; j < block_j; j++)
    {
        typelist[j] = MPI_DOUBLE;
        block_lengths[j] = 1;
        MPI_Get_address(&grid[1][j+1], &displacements[j]);
        displacements[j] = displacements[j] - add_start;
    }

    MPI_Type_create_struct(block_j, block_lengths.data(), displacements.data(), typelist.data(), &send_lower);
    MPI_Type_commit(&send_lower);

    // receive from the block above - allocate to top ghost layer

    MPI_Get_address(&grid[block_i+1][1], &add_start);

    for (int j = 0; j < block_j; j++)
    {
        typelist[j] = MPI_DOUBLE;
        block_lengths[j] = 1;
        MPI_Get_address(&grid[block_i+1][j+1], &displacements[j]);
        displacements[j] = displacements[j] - add_start;
    }

    MPI_Type_create_struct(block_j, block_lengths.data(), displacements.data(), typelist.data(), &recv_upper);
    MPI_Type_commit(&recv_upper);

    // receive from the block below current block - allocate bottom ghost layer

    MPI_Get_address(&grid[0][1], &add_start);

    for (int j = 0; j < block_j; j++)
    {
        typelist[j] = MPI_DOUBLE;
        block_lengths[j] = 1;
        MPI_Get_address(&grid[0][j+1], &displacements[j]);
        displacements[j] = displacements[j] - add_start;
    }

    MPI_Type_create_struct(block_j, block_lengths.data(), displacements.data(), typelist.data(), &recv_lower);
    MPI_Type_commit(&recv_lower);
}

// allocate contiguous memory to store data in decomposed domain over 3 time steps: n+1, n and n-1
void Wave::setup_grids()
{
    // 1D grids
    // originally grid size would be block_i * block_j, but need to include ghost layers surrounding the grid
    grid_1d = new double[(block_i + 2) * (block_j + 2)];
    old_grid_1d = new double[(block_i + 2) * (block_j + 2)];
    new_grid_1d = new double[(block_i + 2) * (block_j + 2)];

    // 2D grids from 1D grids - easier for indexing
    grid = new double*[block_i + 2];
    old_grid = new double*[block_i + 2];
    new_grid = new double*[block_i + 2];

    for (int i = 0; i < block_i + 2; i++)
    {
        grid[i] = &grid_1d[i * (block_j + 2)];
        old_grid[i] = &old_grid_1d[i * (block_j + 2)];
        new_grid[i] = &new_grid_1d[i * (block_j + 2)];
    }

    // fill grid with zeroes apart from ghost layer
    for (int i = 1; i < block_i + 1; i++)
    {
        for (int j = 1; j < block_j + 1; j++)
        {
            grid[i][j] = 0.0;
            old_grid[i][j] = 0.0;
            new_grid[i][j] = 0.0;
        }
    }
}

// deallocate memory for grids
void Wave::free_grids()
{
    delete[] grid;
    delete[] old_grid;
    delete[] new_grid;
    delete[] grid_1d;
    delete[] old_grid_1d;
    delete[] new_grid_1d;
}

// break up domain into blocks based on number of processes p
void Wave::domain_decomposition()
{     
    // adapted from worksheet 2 exercise 3
    for (int i = round(sqrt(p)); i >= 1; i--)
    {
        if (p % i == 0) 
        {
            i_num = i;
            break;
        }
    }

    j_num = p / i_num;

    // number of grids in i direction in the block
    int i_grid = (int) imax / i_num; 
    // remainder from division
    int i_remainder = imax % i_num;

    // number of grids in j direction in the block
    int j_grid = (int) jmax / j_num; 
    // remainder from division
    int j_remainder = jmax % j_num;

    // allocate block for each process over column then row 
    // e.g. 6 processes, domain is divied into 2 x 3 blocks, (0,0) is allocated to process 0, (0,1) to process 1,
    // (0,2) to process 2, (1,0) to process 3, (1,1) to process 4, (1,2) to process 5
   
    i_start = (int) id / j_num * i_grid;
    // add remainder to uppermost row
    if (id >= (i_num * (i_num - 1))) i_end = i_start + i_grid + i_remainder;
    else i_end = i_start + i_grid;

    j_start = (int) (id % j_num) * j_grid;
    // add remainder to right edge column
    if (id % j_num == j_remainder || id == (j_num - 1)) j_end = j_start + j_grid + j_remainder;
    else j_end = j_start + j_grid;

    // dimension of the block in i and j direction
    block_i = i_end - i_start;
    block_j = j_end - j_start;
}



// locate neighbour ids and stencil ids
void Wave::find_neighbours()
{
    // adapted from worksheet 2 exercise 3
    // find blocks that are next to current block vertically, horizontally and diagonally
    // store the ids of processes responsible for those blocks
    
    int id_i = id % i_num;
    int id_j = id / i_num;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int neighbour_i = id_i + i, neighbour_j = id_j + j;

            if (neighbour_i >= 0 && neighbour_i < i_num && neighbour_j >= 0 && neighbour_j < j_num)
            {
                int neighbour_id = neighbour_i + neighbour_j * i_num;
                if (neighbour_id != id) neighbour_ids.push_back(neighbour_id);
            }
        }
    }

    // find stencil blocks and store their ids
    // stnecils are the blocks vertically above and below, and horizontally left and right of current block

    int test_ids[4] = {id-1, id+1, id+j_num, id-j_num};

    for (int i = 0; i < 4; i++)
    {
        for (int neighbour_id : neighbour_ids)
        {
            if (neighbour_id == test_ids[i])
                stencil_ids.push_back(neighbour_id);
        }
    }
}

// generate initial disturbance, adapted from serial wave equation
void Wave::initial_disturbance(double r_splash, double x_splash, double y_splash)
{
    for (int i = 1; i < imax - 1; i++)
    {
        for (int j = 1; j < jmax - 1; j++)
		{
			double x = dx * i;
			double y = dy * j;

			double dist = sqrt(pow(x - x_splash, 2.0) + pow(y - y_splash, 2.0));

			if (dist < r_splash)
			{
				double h = 5.0*(cos(dist / r_splash * M_PI) + 1.0);

                // write value to block if in range of block relative to whole domain
                if (i >= i_start && i < i_end && j >= j_start && j < j_end)
                {
                    grid[i - i_start + 1][j - j_start + 1] = h;
                    old_grid[i - i_start + 1][j - j_start + 1] = h;
                }
			}
		}
    }
}

// write grid data to file, adapted from Serial Wave Equation
void Wave::grid_to_file()
{
    // store filename in a specific format: output_id_timestep.txt

    /* adapted from https://stackoverflow.com/questions/29200635/convert-float-to-string-with-precision-number-of-decimal-digits-specified */
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << t; // for time step store two decimal places in file name
    std::string t_str = stream.str();
    
    std::string fname = "output_" + std::to_string(id) + "_" + t_str + ".txt";

    std::fstream data_out;

    data_out.open(fname, std::fstream::out | std::fstream::app);
    if (data_out.fail())
        throw std::runtime_error("Failed to open file to write grid data");

    for (int i = 1; i < block_i + 1; i++)
    {
        for (int j = 1; j < block_j + 1; j++)
            data_out << grid[i][j] << " ";
        data_out << std::endl;
    }
    data_out.close();
}

// write start and end points of block in terms of i and j to file - each process writes to same file
void Wave::parameters_to_file()
{
    MPI_File mpi_file;
    
    std::string fname = "parameters.txt";

    if (MPI_File_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &mpi_file) != MPI_SUCCESS)
	    throw std::runtime_error("Error opening file to write parameters");

    std::string out_str;

    out_str = std::to_string(i_start) + "\t" + std::to_string(i_end) + "\t" + std::to_string(j_start) + "\t" + std::to_string(j_end);
    out_str += "\n";

    MPI_File_write_ordered(mpi_file, out_str.c_str(), out_str.size(), MPI_BYTE, MPI_STATUS_IGNORE);
	MPI_File_close(&mpi_file);
}

// write data about the whole domain, number of processes, maximum time step, time step size to file
void Wave::domain_to_file()
{
    if (id == 0)
    {
        std::string fname = "domain.txt";

        std::fstream data_out;

        data_out.open(fname, std::fstream::out | std::fstream::app);
        if (data_out.fail())
            throw std::runtime_error("Cannot open file to write grid data");

        data_out << imax << "\t" << jmax << "\t" << y_max << "\t" << x_max << "\t" << c << "\t" << p << "\t" << dt_out << "\t" << t_max << "\t" << dt << std::endl;
        data_out.close();
    }
}

// send data to and receive data from stencils about ghost layers
void Wave::get_ghost_layer()
{    
    MPI_Request *request_list = new MPI_Request[stencil_ids.size() * 2];
    int cnt = 0;

    // go over ids of stencils for each process and send data from the appropriate locations in grid using mpi datatypes
    for (int stencil_id : stencil_ids)
    {
        // send second column to left stencil
        if (stencil_id == id - 1)
        {
            MPI_Isend(&grid[1][1], 1, send_left, stencil_id, tag_num, MPI_COMM_WORLD, &request_list[cnt]);
            cnt++;
        }
        // send second last column to right stencil
        else if (stencil_id == id + 1)
        {
            MPI_Isend(&grid[1][block_j], 1, send_right, stencil_id, tag_num, MPI_COMM_WORLD, &request_list[cnt]);
            cnt++;
        }
        // send second last row to stencil above
        else if (stencil_id == id + j_num)
        {
            MPI_Isend(&grid[block_i][1], 1, send_upper, stencil_id, tag_num, MPI_COMM_WORLD, &request_list[cnt]);
            cnt++;
        }
        // send second row to stencil below
        else if (stencil_id == id - j_num)
        {
            MPI_Isend(&grid[1][1], 1, send_lower, stencil_id, tag_num, MPI_COMM_WORLD, &request_list[cnt]);
            cnt++;
        }
    }

    // go over stencil ids for each process, receive from those processes and allocate the data appropriately in the grid using mpi datatypes
    for (int stencil_id : stencil_ids)
    {
        // receive from left stencil, store in first column as ghost layer
        if (stencil_id == id - 1)
        {
            MPI_Irecv(&grid[1][0], 1, recv_left, stencil_id, tag_num, MPI_COMM_WORLD, &request_list[cnt]);
            cnt++;
        }
        // receive from right stencil, store in last column as ghost layer
        else if (stencil_id == id + 1)
        {
            MPI_Irecv(&grid[1][block_j+1], 1, recv_right, stencil_id, tag_num, MPI_COMM_WORLD, &request_list[cnt]);
            cnt++;
        }
        // receive from stencil above, store in last row as ghost layer
        else if (stencil_id == id + j_num)
        {
            MPI_Irecv(&grid[block_i+1][1], 1, recv_upper, stencil_id, tag_num, MPI_COMM_WORLD, &request_list[cnt]);
            cnt++;
        }
        // receive from stencil below, store in first row as ghost layer
        else if (stencil_id == id - j_num)
        {
            MPI_Irecv(&grid[0][1], 1, recv_lower, stencil_id, tag_num, MPI_COMM_WORLD, &request_list[cnt]);
            cnt++;
        }
    }

    // ensure peer-to-peer communication is complete before proceeding
    MPI_Waitall(cnt, request_list, MPI_STATUSES_IGNORE);

    delete[] request_list;
}

// solve wave equation
void Wave::do_iteration()
{   
    // solve for all grids using ghost layers
    for (int i = 1; i < block_i + 1; i++)
		for (int j = 1; j < block_j + 1; j++)
			new_grid[i][j] = pow(dt * c, 2.0) * ((grid[i + 1][j] - 2.0 * grid[i][j] + grid[i - 1][j]) / pow(dx, 2.0) \
            + (grid[i][j + 1] - 2.0 * grid[i][j] + grid[i][j - 1]) / pow(dy, 2.0)) + 2.0 * grid[i][j] - old_grid[i][j];


    // Neumann boundary condition
    if (bc == 'n')
    {   
        // bottom left 
        if (id == 0)
        {   
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][1] = new_grid[i][2];

            for (int j = 1; j < block_j + 1; j++)
                new_grid[1][j] = new_grid[2][j];
        }

        // bottom right 
        else if (id == j_num - 1)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][block_j] = new_grid[i][block_j - 1];

            for (int j = 1; j < block_j + 1; j++)
                new_grid[1][j] = new_grid[2][j];
        }

        // top left 
        else if (id == p - j_num)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][1] = new_grid[i][2];

            for (int j = 1; j < block_j + 1; j++)
                new_grid[block_i][j] = new_grid[block_i - 1][j];
        }

        // top right 
        else if (id == p - 1)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][block_j] = new_grid[i][block_j - 1];

            for (int j = 1; j < block_j + 1; j++)
                new_grid[block_i][j] = new_grid[block_i - 1][j];
        }
        
        // left boundary except top left or bottom left
        else if (j_start == 0)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][1] = new_grid[i][2];
        }

        // right boundary except top right or bottom right
        else if (j_end == jmax)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][block_j] = new_grid[i][block_j - 1];
        }

        // bottom boundary except bottom left or bottom right
        else if (i_start == 0)
        {
            for (int j = 1; j < block_j + 1; j++)
                new_grid[1][j] = new_grid[2][j];
        }

        // top boundary except top left or top right
        else if (i_end == imax)
        {
            for (int j = 1; j < block_j + 1; j++)
                new_grid[block_i][j] = new_grid[block_i - 1][j];
        }
    }

    // Dirichlet boundary condition - have set u = 0
    else if (bc == 'd')
    {
        // bottom left 
        if (id == 0)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][1] = 0.0;

            for (int j = 1; j < block_j + 1; j++)
                new_grid[1][j] = 0.0;
        }
        // bottom right
        else if (id == j_num - 1)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][block_j] = 0.0;

            for (int j = 1; j < block_j + 1; j++)
                new_grid[1][j] = 0.0;
        }
        // top left
        else if (id == p - j_num)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][1] = 0.0;

            for (int j = 1; j < block_j + 1; j++)
                new_grid[block_i][j] = 0.0;
        }
        // top right
        else if (id == p - 1)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][block_j] = 0.0;

            for (int j = 1; j < block_j + 1; j++)
                new_grid[block_i][j] = 0.0;
        }
        
        // left boundary except bottom left or top left
        else if (j_start == 0)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][1] = 0.0;
        }

        // right boundary except bottom right or top right
        else if (j_end == jmax)
        {
            for (int i = 1; i < block_i + 1; i++)
                new_grid[i][block_j] = 0.0;
        }

        // bottom boundary except bottom left or bottom right
        else if (i_start == 0)
        {
            for (int j = 1; j < block_j + 1; j++)
                new_grid[1][j] = 0.0;
        }

        // top boundary except top left or top right
        else if (i_end == imax)
        {
            for (int j = 1; j < block_j + 1; j++)
                new_grid[block_i][j] = 0.0;
        }
    }
    
    // increment by timestep
    t += dt;

    // current grid becomes old grid, new grid becomes current grid
    std::swap(old_grid, new_grid);
    std::swap(old_grid, grid);
    std::swap(old_grid_1d, new_grid_1d);
    std::swap(old_grid_1d, grid_1d);
}
