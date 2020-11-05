simulation
{
}
!--------the node configuration for this simulation--------
node_conf
{
    node_number(1:2) = 1,1,
    if_periodic(1:2) = .true.,.true.,
    n_threads=4,
}
!----------spatial grid----------
grid
{
    nx_p(1:2) = 512,256,
    coordinates = "cartesian",
}
!----------time step and global data dump timestep number----------
time_step
{
    dt = 0.025,
    ndump = 1,
}
restart
{
    ndump_fac = 0,
    if_restart=.false.,
! if_remold=.true.,
}
!----------spatial limits of the simulations----------
!(note that this includes information about
! the motion of the simulation box)
space
{
    xmin(1:2) = 0.000e0,0.000e0,
    xmax(1:2) = 51.2e0, 25.6e0,
    if_move(1:2) = .false.,.false.,
}
!----------time limits ----------
time
{
    tmin = 0.0e0,
    tmax = 150.0e0,
}
!----------field solver set up-----------------
el_mag_fld
{
    solver = "fei", 
}
!----------boundary conditions for em-fields ----------
emf_bound
{
}
emf_solver
{
}

diag_emf
{
    ndump_fac = 100,
    ndump_fac_ene_int = 1,
    reports = "e1", "e2", "e3", "b1", "b2", "b3",     
}

!----------number of particle species----------
particles
{
    num_species = 2,
    interpolation = "linear",
}

!----------information for ELECTRONS----------
species
{
    name = "electrons",
    rqm = -1.0,
    q_real = -1.0,
    num_par_x(1:2) = 2,2,
    push_type = "standard",
}
udist
{
    uth(1:3) = 0.0001 , 0.0001 , 0.0001 ,
    ufl(1:3) = 50.0 , 0.0 , 0.0 ,
}
profile
{
    density = 10,
    num_x = 2,
    fx(1:2,1) = 1, 1,
    x(1:2,1) = 0.000, 100.0,
    fx(1:2,2) = 1, 1,
    x(1:2,2) = 0.000, 100,
}
!----------boundary conditions for this species----------
spe_bound
{

}
diag_species
{
    ndump_fac = 1,
    ndump_fac_ene = 1, 
    reports = "charge",
    ndump_fac_pha = 1,
    ps_pmin(1:3) = 0, -1.3, -1.3,
    ps_pmax(1:3) = 60, 1.3, 1.3,
    phasespaces = "p1x1", 
}


species
{
    name = "ions",
    free_stream = .true.,
    ! num_par_max = 250000,
    rqm = 100.0,
    q_real = 1.0,
    num_par_x(1:2) = 2,2,
}
udist
{
    uth(1:3) = 0.0 , 0.0 , 0.0 ,
    ufl(1:3) = 50.0 , 0.0 , 0.0 ,
}
profile
{
    num_x = 2,
    fx(1:2,1) = 1, 1,
    x(1:2,1) = 0.000, 100.0,
    fx(1:2,2) = 1, 1,
    x(1:2,2) = 0.000, 100,
}
!----------boundary conditions for this species----------
spe_bound
{

}
diag_species
{
    ! ndump_fac_pha = 1,
}



!----------SMOOTHING FOR CURRENTS------------------------------------------------------------
smooth
{
    type = "none",
}

diag_current
{
    ndump_fac = 0,
}


