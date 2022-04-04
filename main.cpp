#include "Parallel_Wave_Equation.cpp"

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* initialise object to use class member functions */
    Wave wave;  

    wave.domain_decomposition();

    wave.find_neighbours();

    wave.setup_grids();

    dx = x_max / ((double)jmax - 1);
	dy = y_max / ((double)imax - 1);

    /* find start and end position of blocks in terms of x and y */
    wave.y_start = dy * wave.i_start;
    wave.y_end = dy * wave.i_end;

    wave.x_start = dx * wave.j_start;
    wave.x_end = dx * wave.j_end;

	t = 0.0;

    /* timestep constraint */
	dt = 0.1 * min(dx, dy) / c;

	int out_cnt = 0, it = 0;

    /* sets half sinusoidal intitial disturbance - adapted from Serial Wave Equation */

	double r_splash = 1.0;
	double x_splash = 3.0;
	double y_splash = 3.0;

    wave.initial_disturbance(r_splash, x_splash, y_splash);

    /* initialise mpi datatypes for peer-to-peer communication */
    wave.buildMPITypes();


    out_cnt++;
	t_out += dt_out;


    /* write to data to files for post processing */

    wave.domain_to_file();
    wave.parameters_to_file();
    wave.grid_to_file();


    /* start timing */

#ifdef DO_TIMING
	auto start = chrono::high_resolution_clock::now();
#endif

    /* solve wave equation until time reaches maximum time step */

    while (t < t_max)
    {
        wave.get_ghost_layer();

        wave.do_iteration();

        /* save grid data to file every dt_out seconds */
        if (t_out <= t)
		{
			// cout << "output: " << out_cnt << "\tt: " << t << "\titeration: " << it << endl;
            wave.grid_to_file();
			out_cnt++;
			t_out += dt_out;
		}

		it++;
    }

    /* ensure all processes completed calculations */
    MPI_Barrier(MPI_COMM_WORLD);

    /* stop timing and print out run time */

#ifdef DO_TIMING
	auto finish = chrono::high_resolution_clock::now();
	if (id == 0)
	{
		std::chrono::duration<double> elapsed = finish - start;
		cout << setprecision(5);
		cout << "The code took " << elapsed.count() << "s to run" << endl;
	}
#endif

    /* deallocate memory */
    
    MPI_Type_free(&wave.send_left);
    MPI_Type_free(&wave.send_right);
    MPI_Type_free(&wave.send_upper);
    MPI_Type_free(&wave.send_lower);
    MPI_Type_free(&wave.recv_left);
    MPI_Type_free(&wave.recv_right);
    MPI_Type_free(&wave.recv_upper);
    MPI_Type_free(&wave.recv_lower);

    wave.free_grids();


    /* tidy up */

    if (id == 0)
    {
        system("mkdir output");
        system("mv *.txt ./output/");
    }


    MPI_Finalize();

    return 0;
}