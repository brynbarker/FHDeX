
#include "hydro_test_functions.H"
#include "hydro_test_functions_F.H"

#include "hydro_functions.H"

#include "StochMomFlux.H"

#ifndef AMREX_USE_CUDA
#include "StructFact.H"
#endif

#include "rng_functions_F.H"

#include "common_functions.H"

#include "gmres_functions.H"

#include "common_namespace_declarations.H"

#include "gmres_namespace_declarations.H"

#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

// argv contains the name of the inputs file entered at the command line
void main_driver(const char* argv)
{

    BL_PROFILE_VAR("main_driver()",main_driver);

    // store the current time so we can later compute total run time.
    Real strt_time = ParallelDescriptor::second();

    std::string inputs_file = argv;

    // read in parameters from inputs file into F90 modules
    // we use "+1" because of amrex_string_c_to_f expects a null char termination
    read_common_namelist(inputs_file.c_str(),inputs_file.size()+1);
    read_gmres_namelist(inputs_file.c_str(),inputs_file.size()+1);

    // copy contents of F90 modules to C++ namespaces
    InitializeCommonNamespace();
    InitializeGmresNamespace();

    // is the problem periodic?
    Vector<int> is_periodic(AMREX_SPACEDIM,0);  // set to 0 (not periodic) by default
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        if (bc_vel_lo[i] == -1 && bc_vel_hi[i] == -1) {
            is_periodic[i] = 1;
        }
    }

    // make BoxArray and Geometry
    BoxArray ba;
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(           0,            0,            0));
        IntVect dom_hi(AMREX_D_DECL(n_cells[0]-1, n_cells[1]-1, n_cells[2]-1));
        Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(domain);

        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        // note we are converting "Vector<int> max_grid_size" to an IntVect
        ba.maxSize(IntVect(max_grid_size));

       // This defines the physical box, [-1,1] in each direction.
        RealBox real_box({AMREX_D_DECL(prob_lo[0],prob_lo[1],prob_lo[2])},
                         {AMREX_D_DECL(prob_hi[0],prob_hi[1],prob_hi[2])});

        // This defines a Geometry object
        geom.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());
    }

    Real dt = fixed_dt;
    Real dtinv = 1.0/dt;
    const Real* dx = geom.CellSize();

    // how boxes are distrubuted among MPI processes
    DistributionMapping dmap(ba);

    /////////////////////////////////////////
    //Initialise rngs
    /////////////////////////////////////////
    const int n_rngs = 1;

    if (restart <= 0) {
        int fhdSeed = 1;
        int particleSeed = 2;
        int selectorSeed = 3;
        int thetaSeed = 4;
        int phiSeed = 5;
        int generalSeed = 6;

        //Initialise rngs
        rng_initialize(&fhdSeed,&particleSeed,&selectorSeed,&thetaSeed,&phiSeed,&generalSeed);
    }
    /////////////////////////////////////////

    ///////////////////////////////////////////
    // rho, alpha, beta, gamma:
    ///////////////////////////////////////////

    MultiFab rho(ba, dmap, 1, 1);
    rho.setVal(1.);

    // alpha_fc arrays
    std::array< MultiFab, AMREX_SPACEDIM > alpha_fc;
    AMREX_D_TERM(alpha_fc[0].define(convert(ba,nodal_flag_x), dmap, 1, 1);,
                 alpha_fc[1].define(convert(ba,nodal_flag_y), dmap, 1, 1);,
                 alpha_fc[2].define(convert(ba,nodal_flag_z), dmap, 1, 1););
    AMREX_D_TERM(alpha_fc[0].setVal(dtinv);,
                 alpha_fc[1].setVal(dtinv);,
                 alpha_fc[2].setVal(dtinv););

    // beta cell centred
    MultiFab beta(ba, dmap, 1, 1);
    beta.setVal(visc_coef);

    // beta on nodes in 2d
    // beta on edges in 3d
    std::array< MultiFab, NUM_EDGE > beta_ed;
#if (AMREX_SPACEDIM == 2)
    beta_ed[0].define(convert(ba,nodal_flag), dmap, 1, 1);
    beta_ed[0].setVal(visc_coef);
#elif (AMREX_SPACEDIM == 3)
    beta_ed[0].define(convert(ba,nodal_flag_xy), dmap, 1, 1);
    beta_ed[1].define(convert(ba,nodal_flag_xz), dmap, 1, 1);
    beta_ed[2].define(convert(ba,nodal_flag_yz), dmap, 1, 1);
    beta_ed[0].setVal(visc_coef);
    beta_ed[1].setVal(visc_coef);
    beta_ed[2].setVal(visc_coef);
#endif

    // cell-centered gamma
    MultiFab gamma(ba, dmap, 1, 1);
    gamma.setVal(0.);

    ///////////////////////////////////////////

    ///////////////////////////////////////////
    // Define & initalize eta & temperature multifabs
    ///////////////////////////////////////////
    // eta & temperature
    const Real eta_const = visc_coef;
    const Real temp_const = T_init[0];      // [units: K]

    // eta & temperature cell centered
    MultiFab  eta_cc;
    MultiFab temp_cc;
    // eta & temperature nodal
    std::array< MultiFab, NUM_EDGE >   eta_ed;
    std::array< MultiFab, NUM_EDGE >  temp_ed;
    // eta cell-centered
    eta_cc.define(ba, dmap, 1, 1);
    // temperature cell-centered
    temp_cc.define(ba, dmap, 1, 1);
#if (AMREX_SPACEDIM == 2)
    // eta nodal
    eta_ed[0].define(convert(ba,nodal_flag), dmap, 1, 0);
    // temperature nodal
    temp_ed[0].define(convert(ba,nodal_flag), dmap, 1, 0);
#elif (AMREX_SPACEDIM == 3)
    // eta nodal
    eta_ed[0].define(convert(ba,nodal_flag_xy), dmap, 1, 0);
    eta_ed[1].define(convert(ba,nodal_flag_xz), dmap, 1, 0);
    eta_ed[2].define(convert(ba,nodal_flag_yz), dmap, 1, 0);
    // temperature nodal
    temp_ed[0].define(convert(ba,nodal_flag_xy), dmap, 1, 0);
    temp_ed[1].define(convert(ba,nodal_flag_xz), dmap, 1, 0);
    temp_ed[2].define(convert(ba,nodal_flag_yz), dmap, 1, 0);
#endif

    // Initalize eta & temperature multifabs
    // eta cell-centered
    eta_cc.setVal(eta_const);
    // temperature cell-centered
    temp_cc.setVal(temp_const);
#if (AMREX_SPACEDIM == 2)
    // eta nodal
    eta_ed[0].setVal(eta_const);
    // temperature nodal
    temp_ed[0].setVal(temp_const);
#elif (AMREX_SPACEDIM == 3)
    // eta nodal
    eta_ed[0].setVal(eta_const);
    eta_ed[1].setVal(eta_const);
    eta_ed[2].setVal(eta_const);
    // temperature nodal
    temp_ed[0].setVal(temp_const);
    temp_ed[1].setVal(temp_const);
    temp_ed[2].setVal(temp_const);
#endif
    ///////////////////////////////////////////

    ///////////////////////////////////////////
    // random fluxes:
    ///////////////////////////////////////////

    // mflux divergence, staggered in x,y,z

    // Define mfluxdiv predictor multifabs
    std::array< MultiFab, AMREX_SPACEDIM >  mfluxdiv_stoch;
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
      mfluxdiv_stoch[d].define(convert(ba,nodal_flag_dir[d]), dmap, 1, 0);
      mfluxdiv_stoch[d].setVal(0.0);
    }

    Vector< amrex::Real > weights;
    // weights = {std::sqrt(0.5), std::sqrt(0.5)};
    weights = {1.0};

    // Declare object of StochMomFlux class
    StochMomFlux sMflux (ba,dmap,geom,n_rngs);

#ifndef AMREX_USE_CUDA
    ///////////////////////////////////////////
    // Initialize structure factor object for analysis
    ///////////////////////////////////////////
    
    // variables are velocities
    int structVars = AMREX_SPACEDIM;
    
    Vector< std::string > var_names;
    var_names.resize(structVars);
    
    int cnt = 0;
    std::string x;

    // velx, vely, velz
    for (int d=0; d<AMREX_SPACEDIM; d++) {
      x = "vel";
      x += (120+d);
      var_names[cnt++] = x;
    }

    MultiFab structFactMF(ba, dmap, structVars, 0);

    // need to use dVol for scaling
    Real dVol = dx[0]*dx[1];
    if (AMREX_SPACEDIM == 2) {
	dVol *= cell_depth;
    } else if (AMREX_SPACEDIM == 3) {
	dVol *= dx[2];
    }
    
    Vector<Real> var_scaling(structVars*(structVars+1)/2);
    for (int d=0; d<var_scaling.size(); ++d) {
        var_scaling[d] = 1./dVol;
    }

#if 1
    // option to compute all pairs
    StructFact structFact(ba,dmap,var_names,var_scaling);
#else
    // option to compute only specified pairs
    int nPairs = 2;
    amrex::Vector< int > s_pairA(nPairs);
    amrex::Vector< int > s_pairB(nPairs);

    // Select which variable pairs to include in structure factor:
    s_pairA[0] = 0;
    s_pairB[0] = 0;
    s_pairA[1] = 1;
    s_pairB[1] = 1;
    
    StructFact structFact(ba,dmap,var_names,var_scaling,s_pairA,s_pairB);
#endif
    
#endif

    ///////////////////////////////////////////
    
    // FIXME need to fill physical boundary condition ghost cells for tracer

    // pressure for GMRES solve
    MultiFab pres(ba,dmap,1,1);
    pres.setVal(0.);  // initial guess

    std::array< MultiFab, AMREX_SPACEDIM > umacNew;
    AMREX_D_TERM(umacNew[0].define(convert(ba,nodal_flag_x), dmap, 1, 1);,
                 umacNew[1].define(convert(ba,nodal_flag_y), dmap, 1, 1);,
                 umacNew[2].define(convert(ba,nodal_flag_z), dmap, 1, 1););   
    
    int step_start;
    amrex::Real time;

    // tracer
    MultiFab tracer(ba,dmap,1,1);
    
    // staggered velocities
    std::array< MultiFab, AMREX_SPACEDIM > umac;

    if (restart > 0) {
        ReadCheckPoint(step_start,time,umac,tracer);
    }
    else {

        tracer.setVal(0.);

        AMREX_D_TERM(umac[0].define(convert(ba,nodal_flag_x), dmap, 1, 1);,
                     umac[1].define(convert(ba,nodal_flag_y), dmap, 1, 1);,
                     umac[2].define(convert(ba,nodal_flag_z), dmap, 1, 1););
    
        const RealBox& realDomain = geom.ProbDomain();
        int dm;

        for ( MFIter mfi(beta); mfi.isValid(); ++mfi ) {
            const Box& bx = mfi.validbox();

            AMREX_D_TERM(dm=0; init_vel(BL_TO_FORTRAN_BOX(bx),
                                        BL_TO_FORTRAN_ANYD(umac[0][mfi]), geom.CellSize(),
                                        geom.ProbLo(), geom.ProbHi() ,&dm,
                                        ZFILL(realDomain.lo()), ZFILL(realDomain.hi()));,
                         dm=1; init_vel(BL_TO_FORTRAN_BOX(bx),
                                        BL_TO_FORTRAN_ANYD(umac[1][mfi]), geom.CellSize(),
                                        geom.ProbLo(), geom.ProbHi() ,&dm,
                                        ZFILL(realDomain.lo()), ZFILL(realDomain.hi()));,
                         dm=2; init_vel(BL_TO_FORTRAN_BOX(bx),
                                        BL_TO_FORTRAN_ANYD(umac[2][mfi]), geom.CellSize(),
                                        geom.ProbLo(), geom.ProbHi() ,&dm,
                                        ZFILL(realDomain.lo()), ZFILL(realDomain.hi())););

    	// initialize tracer
        init_s_vel(BL_TO_FORTRAN_BOX(bx),
    		   BL_TO_FORTRAN_ANYD(tracer[mfi]),
    		   dx, ZFILL(realDomain.lo()), ZFILL(realDomain.hi()));

    }
    
        // Add initial equilibrium fluctuations
        if(initial_variance_mom != 0.0) {
            addMomFluctuations(umac, rho, temp_cc, initial_variance_mom,geom);
        }

        // Project umac onto divergence free field
        MultiFab macrhs(ba,dmap,1,1);
        macrhs.setVal(0.0);
        MacProj_hydro(umac,rho,geom,true);

        step_start = 1;
        time = 0.;

#ifndef AMREX_USE_CUDA        
        // We do the analysis first so we include the initial condition in the files if n_steps_skip=0
        if (n_steps_skip == 0 && struct_fact_int > 0) {

            // add this snapshot to the average in the structure factor

            // copy velocities into structFactMF
            for(int d=0; d<AMREX_SPACEDIM; d++) {
                ShiftFaceToCC(umac[d], 0, structFactMF, d, 1);
            }
            structFact.FortStructure(structFactMF,geom);
        }
#endif

        // write out initial state
        // write out umac, tracer, pres, and divergence to a plotfile
        if (plot_int > 0) {
            WritePlotFile(step_start,time,geom,umac,tracer,pres);
#ifndef AMREX_USE_CUDA
            if (n_steps_skip == 0 && struct_fact_int > 0) {
                structFact.WritePlotFile(0,0.,geom,"plt_SF");
            }
#endif
        }

    }

    ///////////////////////////////////////////   

    // initial guess for new solution
    for (int i=0; i<AMREX_SPACEDIM; i++) {
      MultiFab::Copy(umacNew[i], umac[i], 0, 0, 1, 0);
    }
    //Time stepping loop
    for(int step=step_start;step<=max_step;++step) {

        Real step_strt_time = ParallelDescriptor::second();

	if(variance_coef_mom != 0.0) {

	  // Fill stochastic terms
	  sMflux.fillMomStochastic();

	  // compute stochastic force terms
	  sMflux.StochMomFluxDiv(mfluxdiv_stoch,0,eta_cc,eta_ed,temp_cc,temp_ed,weights,dt);
	}

	// Advance umac
	advance(umac,umacNew,pres,tracer,mfluxdiv_stoch,
		alpha_fc,beta,gamma,beta_ed,geom,dt);

	//////////////////////////////////////////////////

#ifndef AMREX_USE_CUDA
	if (step > n_steps_skip && struct_fact_int > 0 && (step-n_steps_skip)%struct_fact_int == 0) {

            // add this snapshot to the average in the structure factor

            // copy velocities into structFactMF
            for(int d=0; d<AMREX_SPACEDIM; d++) {
                ShiftFaceToCC(umac[d], 0, structFactMF, d, 1);
            }
            structFact.FortStructure(structFactMF,geom);
        }
#endif
        
        Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        time = time + dt;

        if (plot_int > 0 && step%plot_int == 0) {
            // write out umac, tracer, pres, and divergence to a plotfile
            WritePlotFile(step,time,geom,umac,tracer,pres);
#ifndef AMREX_USE_CUDA
            if (step > n_steps_skip && struct_fact_int > 0) {
                structFact.WritePlotFile(step,time,geom,"plt_SF");
            }
#endif
        }

        if (chk_int > 0 && step%chk_int == 0) {
            // write out umac and tracer to a checkpoint file
            WriteCheckPoint(step,time,umac,tracer);
        }
        
    }

    // Call the timer again and compute the maximum difference between the start time
    // and stop time over all processors
    Real stop_time = ParallelDescriptor::second() - strt_time;
    ParallelDescriptor::ReduceRealMax(stop_time);
    amrex::Print() << "Run time = " << stop_time << std::endl;

}
