#include "main_driver.H"
#include "main_driver_F.H"

#include "hydro_functions.H"
#include "hydro_functions_F.H"

//#include "analysis_functions_F.H"
#include "StochMFlux.H"
//#include "StructFact.H"

#include "rng_functions_F.H"

#include "common_functions.H"
#include "common_functions_F.H"

#include "gmres_functions.H"
#include "gmres_functions_F.H"

#include <ib_functions.H>

#include "common_namespace.H"
#include "common_namespace_declarations.H"

#include <immbdy_namespace.H>
#include <immbdy_namespace_declarations.H>

#include "gmres_namespace.H"
#include "gmres_namespace_declarations.H"

#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>

#include <IBMarkerContainer.H>
#include <IBMarkerMD.H>


using namespace amrex;
using namespace common;
using namespace gmres;
using namespace immbdy_md;
using namespace immbdy;
using namespace ib_flagellum;


//! Defines staggered MultiFab arrays (BoxArrays set according to the
//! nodal_flag_[x,y,z]). Each MultiFab has 1 component, and 1 ghost cell
inline void defineFC(std::array< MultiFab, AMREX_SPACEDIM > & mf_in,
                     const BoxArray & ba, const DistributionMapping & dm,
                     int nghost) {

    for (int i=0; i<AMREX_SPACEDIM; i++)
        mf_in[i].define(convert(ba, nodal_flag_dir[i]), dm, 1, nghost);
}

inline void defineEdge(std::array< MultiFab, AMREX_SPACEDIM > & mf_in,
                     const BoxArray & ba, const DistributionMapping & dm,
                     int nghost) {

    for (int i=0; i<AMREX_SPACEDIM; i++)
        mf_in[i].define(convert(ba, nodal_flag_edge[i]), dm, 1, nghost);
}


//! Sets the value for each component of staggered MultiFab
inline void setVal(std::array< MultiFab, AMREX_SPACEDIM > & mf_in,
                   Real set_val) {

    for (int i=0; i<AMREX_SPACEDIM; i++)
        mf_in[i].setVal(set_val);
}



// argv contains the name of the inputs file entered at the command line
void main_driver(const char * argv) {

    BL_PROFILE_VAR("main_driver()",main_driver);


    /****************************************************************************
     *                                                                          *
     * Initialize simulation                                                    *
     *                                                                          *
     ***************************************************************************/

    // store the current time so we can later compute total run time.
    Real strt_time = ParallelDescriptor::second();


    //___________________________________________________________________________
    // Load parameters from inputs file, and initialize global parameters
    std::string inputs_file = argv;

    // read in parameters from inputs file into F90 modules NOTE: we use "+1"
    // because of amrex_string_c_to_f expects a null char termination
    read_common_namelist(inputs_file.c_str(), inputs_file.size()+1);
    read_gmres_namelist(inputs_file.c_str(), inputs_file.size()+1);

    // copy contents of F90 modules to C++ namespaces NOTE: any changes to
    // global settings in fortran/c++ after this point need to be synchronized
    InitializeCommonNamespace();
    InitializeGmresNamespace();
    InitializeImmbdyNamespace();
    InitializeIBFlagellumNamespace();


    //___________________________________________________________________________
    // Set boundary conditions

    // is the problem periodic?
    Vector<int> is_periodic(AMREX_SPACEDIM,0);  // set to 0 (not periodic) by default
    for (int i=0; i<AMREX_SPACEDIM; ++i)
        if (bc_lo[i] <= -1 && bc_hi[i] <= -1)
            is_periodic[i] = 1;

    //___________________________________________________________________________
    // Make BoxArray, DistributionMapping, and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(             0,              0,              0));
        IntVect dom_hi(AMREX_D_DECL(n_cells[0] - 1, n_cells[1] - 1, n_cells[2] - 1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);

        // Break up boxarray "ba" into chunks no larger than "max_grid_size"
        // along a direction note we are converting "Vector<int> max_grid_size"
        // to an IntVect
        ba.maxSize(IntVect(max_grid_size));

        // This defines the physical box, [-1, 1] in each direction
        RealBox real_box({AMREX_D_DECL(prob_lo[0], prob_lo[1], prob_lo[2])},
                         {AMREX_D_DECL(prob_hi[0], prob_hi[1], prob_hi[2])});

        // This defines a Geometry object
        geom.define(domain, & real_box, CoordSys::cartesian, is_periodic.data());
    }

    // how boxes are distrubuted among MPI processes
    DistributionMapping dmap(ba);


    //___________________________________________________________________________
    // Cell size, and time step
    Real dt         = fixed_dt;
    Real dtinv      = 1.0 / dt;
    const Real * dx = geom.CellSize();


    //___________________________________________________________________________
    // Initialize random number generators
    const int n_rngs = 1;

    // this seems really random :P
    int fhdSeed      = 1;
    int particleSeed = 2;
    int selectorSeed = 3;
    int thetaSeed    = 4;
    int phiSeed      = 5;
    int generalSeed  = 6;

    // each CPU gets a different random seed
    const int proc = ParallelDescriptor::MyProc();
    fhdSeed      += proc;
    particleSeed += proc;
    selectorSeed += proc;
    thetaSeed    += proc;
    phiSeed      += proc;
    generalSeed  += proc;

    // initialize rngs
    rng_initialize( & fhdSeed, & particleSeed, & selectorSeed,
                    & thetaSeed, & phiSeed, & generalSeed);



    /****************************************************************************
     *                                                                          *
     * Initialize physical parameters                                           *
     *                                                                          *
     ***************************************************************************/

    //___________________________________________________________________________
    // Set rho, alpha, beta, gamma:

    // rho is cell-centered
    MultiFab rho(ba, dmap, 1, 1);
    rho.setVal(1.);

    // alpha_fc is face-centered
    Real theta_alpha = 1.;
    std::array< MultiFab, AMREX_SPACEDIM > alpha_fc;
    defineFC(alpha_fc, ba, dmap, 1);
    setVal(alpha_fc, dtinv);

    // beta is cell-centered
    MultiFab beta(ba, dmap, 1, 1);
    beta.setVal(visc_coef);

    // beta is on nodes in 2D, and is on edges in 3D
    std::array< MultiFab, NUM_EDGE > beta_ed;
#if (AMREX_SPACEDIM == 2)
    beta_ed[0].define(convert(ba, nodal_flag), dmap, 1, 1);
    beta_ed[0].setVal(visc_coef);
#elif (AMREX_SPACEDIM == 3)
    defineEdge(beta_ed, ba, dmap, 1);
    setVal(beta_ed, visc_coef);
#endif

    // cell-centered gamma
    MultiFab gamma(ba, dmap, 1, 1);
    gamma.setVal(0.);


    //___________________________________________________________________________
    // Define & initialize eta & temperature MultiFabs

    // eta & temperature
    const Real eta_const  = visc_coef;
    const Real temp_const = T_init[0];      // [units: K]


    // NOTE: eta and temperature live on both cell-centers and edges

    // eta & temperature cell centered
    MultiFab  eta_cc(ba, dmap, 1, 1);
    MultiFab temp_cc(ba, dmap, 1, 1);
    // eta & temperature nodal
    std::array< MultiFab, NUM_EDGE >   eta_ed;
    std::array< MultiFab, NUM_EDGE >  temp_ed;

    // eta_ed and temp_ed are on nodes in 2D, and on edges in 3D
#if (AMREX_SPACEDIM == 2)
    eta_ed[0].define(convert(ba,nodal_flag), dmap, 1, 0);
    temp_ed[0].define(convert(ba,nodal_flag), dmap, 1, 0);

    eta_ed[0].setVal(eta_const);
    temp_ed[0].setVal(temp_const);
#elif (AMREX_SPACEDIM == 3)
    defineEdge(eta_ed, ba, dmap, 1);
    defineEdge(temp_ed, ba, dmap, 1);

    setVal(eta_ed, eta_const);
    setVal(temp_ed, temp_const);
#endif

    // eta_cc and temp_cc are always cell-centered
    eta_cc.setVal(eta_const);
    temp_cc.setVal(temp_const);


    //___________________________________________________________________________
    // Define random fluxes mflux (momentum-flux) divergence, staggered in x,y,z

    // mfluxdiv predictor multifabs
    std::array< MultiFab, AMREX_SPACEDIM >  mfluxdiv_predict;
    defineFC(mfluxdiv_predict, ba, dmap, 1);
    setVal(mfluxdiv_predict, 0.);

    // mfluxdiv corrector multifabs
    std::array< MultiFab, AMREX_SPACEDIM >  mfluxdiv_correct;
    defineFC(mfluxdiv_correct, ba, dmap, 1);
    setVal(mfluxdiv_correct, 0.);

    Vector< amrex::Real > weights;
    // weights = {std::sqrt(0.5), std::sqrt(0.5)};
    weights = {1.0};

    // tracer
    MultiFab tracer(ba, dmap, 1,1);
    tracer.setVal(0.);


    //___________________________________________________________________________
    // Define velocities and pressure

    // pressure for GMRES solve
    MultiFab pres(ba, dmap, 1, 1);
    pres.setVal(0.);  // initial guess

    // staggered velocities
    std::array< MultiFab, AMREX_SPACEDIM > umac;
    defineFC(umac, ba, dmap, 1);

    std::array< MultiFab, AMREX_SPACEDIM > umacNew;
    defineFC(umacNew, ba, dmap, 1);


    //___________________________________________________________________________
    // Define structure factor:

    Vector< std::string > var_names;
    var_names.resize(AMREX_SPACEDIM);
    int cnt = 0;
    std::string x;
    for (int d=0; d<var_names.size(); d++) {
        x = "vel";
        x += (120+d);
        var_names[cnt++] = x;
    }

    MultiFab struct_in_cc;
    struct_in_cc.define(ba, dmap, AMREX_SPACEDIM, 0);

    amrex::Vector< int > s_pairA(AMREX_SPACEDIM);
    amrex::Vector< int > s_pairB(AMREX_SPACEDIM);

    // Select which variable pairs to include in structure factor:
    s_pairA[0] = 0;
    s_pairB[0] = 0;
    //
    s_pairA[1] = 1;
    s_pairB[1] = 1;
    //
#if (AMREX_SPACEDIM == 3)
    s_pairA[2] = 2;
    s_pairB[2] = 2;
#endif

    // StructFact structFact(ba, dmap, var_names);
    // StructFact structFact(ba, dmap, var_names, s_pairA, s_pairB);



    /****************************************************************************
     *                                                                          *
     * Set Initial Conditions                                                   *
     *                                                                          *
     ***************************************************************************/

    // Find the optimal number of ghost cells for the IBMarkerContainer

    Real min_dx = dx[0];
    for (int d=1; d<AMREX_SPACEDIM; ++d)
            min_dx = std::min(min_dx, dx[d]);

    int ib_nghost = 8; // min of 8 is a HACK: something large enough but not too large
    for (int i_ib=0; i_ib < n_immbdy; ++i_ib) {

        if (n_marker[i_ib] <= 0) continue;

        int N       = n_marker[i_ib];
        Real L      = ib_flagellum::length[i_ib];
        Real l_link = L/N;

        int min_nghost = 2*l_link/min_dx;
        ib_nghost      = std::max(ib_nghost, min_nghost);
    }

    Print() << "Initializing IBMarkerContainer with "
            << ib_nghost << " ghost cells" << std::endl;

    IBMarkerContainer ib_mc(geom, dmap, ba, ib_nghost);


    for (int i_ib=0; i_ib < n_immbdy; ++i_ib) {

        if (n_marker[i_ib] <= 0) continue;

        int N  = n_marker[i_ib];
        Real L = ib_flagellum::length[i_ib];

        Real l_link = L/N;

        const RealVect & x_0 = offset_0[i_ib];

        Print() << "Initializing flagellum:" << std::endl;
        Print() << "N=      " << N           << std::endl;
        Print() << "L=      " << L           << std::endl;
        Print() << "l_link= " << l_link      << std::endl;
        Print() << "x_0=    " << x_0         << std::endl;

        Vector<RealVect> marker_positions(N);
        //for (int i=0; i<marker_positions.size(); ++i)
        //    marker_positions[i] = RealVect{x_0[0] + i*l_link, x_0[1], x_0[2]};
	
	// This is a realistic initial waveform normalized to length of 0.5
   	marker_positions[0]  = RealVect{0.0500, 0.5000, 0.5};
    	marker_positions[1]  = RealVect{0.0636, 0.4791, 0.5};
    	marker_positions[2]  = RealVect{0.0768, 0.4579, 0.5};
    	marker_positions[3]  = RealVect{0.0899, 0.4367, 0.5};
    	marker_positions[4]  = RealVect{0.1034, 0.4158, 0.5};
    	marker_positions[5]  = RealVect{0.1176, 0.3952, 0.5};
    	marker_positions[6]  = RealVect{0.1324, 0.3752, 0.5};
    	marker_positions[7]  = RealVect{0.1479, 0.3556, 0.5};
    	marker_positions[8]  = RealVect{0.1637, 0.3363, 0.5};
    	marker_positions[9]  = RealVect{0.1797, 0.3172, 0.5};
    	marker_positions[10] = RealVect{0.1954, 0.2978, 0.5};
    	marker_positions[11] = RealVect{0.2104, 0.2780, 0.5};
    	marker_positions[12] = RealVect{0.2244, 0.2573, 0.5};
    	marker_positions[13] = RealVect{0.2366, 0.2356, 0.5};
    	marker_positions[14] = RealVect{0.2462, 0.2126, 0.5};
    	marker_positions[15] = RealVect{0.2523, 0.1884, 0.5};
    	marker_positions[16] = RealVect{0.2542, 0.1636, 0.5};
    	marker_positions[17] = RealVect{0.2508, 0.1389, 0.5};
    	marker_positions[18] = RealVect{0.2414, 0.1159, 0.5};
    	marker_positions[19] = RealVect{0.2259, 0.0965, 0.5};
    	marker_positions[20] = RealVect{0.2049, 0.0833, 0.5};


        Vector<Real> marker_radii(N);
        for (int i=0; i<marker_radii.size(); ++i) marker_radii[i] = 4*l_link;

        ib_mc.InitList(0, marker_radii, marker_positions, i_ib);
    }

    ib_mc.fillNeighbors();
    ib_mc.PrintMarkerData(0);
    BL_PROFILE_VAR_STOP(createmarkers);
 

    //___________________________________________________________________________
    // Initialize velocities (fluid and tracers)

    BL_PROFILE_VAR("main_initalize velocity of marker",markerv);

    const RealBox& realDomain = geom.ProbDomain();
    int dm;

    for ( MFIter mfi(beta); mfi.isValid(); ++mfi ) {
        const Box& bx = mfi.validbox();

        BL_PROFILE_VAR("main_initalize velocity of marker",markerv);
        // initialize velocity
        for (int d=0; d<AMREX_SPACEDIM; ++d)
             init_vel(BL_TO_FORTRAN_BOX(bx),
                     BL_TO_FORTRAN_ANYD(umac[d][mfi]), geom.CellSize(),
                     geom.ProbLo(), geom.ProbHi(), & d,
                     ZFILL(realDomain.lo()), ZFILL(realDomain.hi()));

        BL_PROFILE_VAR_STOP(markerv);

        BL_PROFILE_VAR("main_initialize tracer",tracer);
        // initialize tracer
        init_s_vel(BL_TO_FORTRAN_BOX(bx),
                   BL_TO_FORTRAN_ANYD(tracer[mfi]),
                   dx, ZFILL(realDomain.lo()), ZFILL(realDomain.hi()));
        BL_PROFILE_VAR_STOP(tracer);
    }


    //___________________________________________________________________________
    // Ensure that ICs satisfy BCs
    BL_PROFILE_VAR("main_ensure initilizaction works",ICwork);

    pres.FillBoundary(geom.periodicity());
    MultiFABPhysBC(pres, geom);

    for (int i=0; i<AMREX_SPACEDIM; i++) {
        umac[i].FillBoundary(geom.periodicity());
        MultiFABPhysBCDomainVel(umac[i], i, geom, i);
        MultiFABPhysBCMacVel(umac[i], i, geom, i);
    }
    BL_PROFILE_VAR_STOP(ICwork);


    //___________________________________________________________________________
    // Add random momentum fluctuations

    // Declare object of StochMFlux class
    StochMFlux sMflux (ba, dmap, geom, n_rngs);

    // Add initial equilibrium fluctuations
    sMflux.addMfluctuations(umac, rho, temp_cc, initial_variance_mom, geom);

    // Project umac onto divergence free field
    MultiFab macphi(ba,dmap, 1, 1);
    MultiFab macrhs(ba,dmap, 1, 1);
    macrhs.setVal(0.);
    MacProj(umac, rho, geom, true); // from MacProj_hydro.cpp

    // initial guess for new solution
    for (int d=0; d<AMREX_SPACEDIM; ++d)
        MultiFab::Copy(umacNew[d], umac[d], 0, 0, 1, 1);

    int step = 0;
    Real time = 0.;


    //___________________________________________________________________________
    // Write out initial state
    if (plot_int > 0) {
        WritePlotFile(step, time, geom, umac, tracer, pres, ib_mc);
    }



    /****************************************************************************
     *                                                                          *
     * Advance Time Steps                                                       *
     *                                                                          *
     ***************************************************************************/

    for(step = 1; step <= max_step; ++step) {

        Real step_strt_time = ParallelDescriptor::second();

        //if(variance_coef_mom != 0.0) {

           //___________________________________________________________________
           // Fill stochastic terms

           // sMflux.fillMStochastic();

            // Compute stochastic force terms (and apply to mfluxdiv_*)
           // sMflux.stochMforce(mfluxdiv_predict, eta_cc, eta_ed, temp_cc, temp_ed, weights, dt);
           // sMflux.stochMforce(mfluxdiv_correct, eta_cc, eta_ed, temp_cc, temp_ed, weights, dt);
        // }

        //___________________________________________________________________
        // Advance umac
        advance(umac, umacNew, pres, tracer, ib_mc, mfluxdiv_predict, mfluxdiv_correct,
                alpha_fc, beta, gamma, beta_ed, geom, dt, time);



        //_______________________________________________________________________
        // Update structure factor

        // if (step > n_steps_skip && struct_fact_int > 0 && (step-n_steps_skip-1)%struct_fact_int == 0) {
        //     for(int d=0; d<AMREX_SPACEDIM; d++)
        //         ShiftFaceToCC(umac[d], 0, struct_in_cc, d, 1);
        //     structFact.FortStructure(struct_in_cc,geom);
        // }

        Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        time = time + dt;

        if (plot_int > 0 && step%plot_int == 0) {
           //write out umac & pres to a plotfile
           WritePlotFile(step, time, geom, umac, tracer, pres, ib_mc);
        }
    }

    ///////////////////////////////////////////
    // if (struct_fact_int > 0) {
    //     Real dVol = dx[0]*dx[1];
    //     int tot_n_cells = n_cells[0]*n_cells[1];
    //     if (AMREX_SPACEDIM == 2) {
    //         dVol *= cell_depth;
    //     } else if (AMREX_SPACEDIM == 3) {
    //         dVol *= dx[2];
    //         tot_n_cells = n_cells[2]*tot_n_cells;
    //     }

    //     let rho = 1
    //     Real SFscale = dVol/(k_B*temp_const);
    //     Print() << "Hack: structure factor scaling = " << SFscale << std::endl;

    //     structFact.Finalize(SFscale);
    //     structFact.WritePlotFile(step,time,geom);
    // }

    // Call the timer again and compute the maximum difference between the start
    // time and stop time over all processors
    // Real stop_time = ParallelDescriptor::second() - strt_time;
    // ParallelDescriptor::ReduceRealMax(stop_time);
    // amrex::Print() << "Run time = " << stop_time << std::endl;

}
