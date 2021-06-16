#include "INS_functions.H"
#include "common_namespace_declarations.H"
#include "gmres_namespace_declarations.H"
#include "species.H"
#include "paramPlane.H"
#include "StructFact.H"
#include "particle_functions.H"
#include "chrono"
#include "iostream"
#include "fstream"
#include "DsmcParticleContainer.H"
#include <AMReX_PlotFileUtil.H>

using namespace std::chrono;
using namespace std;
// argv contains the name of the inputs file entered at the command line
void main_driver(const char* argv)
{
	// timer for total simulation time
	Real strt_time = ParallelDescriptor::second();

	std::string inputs_file = argv;

	// read in parameters from inputs file into F90 modules
	// we use "+1" because of amrex_string_c_to_f expects a null char termination
	read_common_namelist(inputs_file.c_str(),inputs_file.size()+1);

	InitializeCommonNamespace();
	InitializeGmresNamespace();

	BoxArray ba;
	IntVect dom_lo(AMREX_D_DECL(           0,            0,            0));
	IntVect dom_hi(AMREX_D_DECL(n_cells[0]-1, n_cells[1]-1, n_cells[2]-1));
	Box domain(dom_lo, dom_hi);
	
	DistributionMapping dmap;
	
	MultiFab cu, cuMeans, cuVars, coVars;
	int nnspec, nstats;
	// Will likely need to redo this
	if (restart < 0) {
		if (seed > 0) {
			InitRandom(seed+ParallelDescriptor::MyProc());
		} else if (seed == 0) {
			auto now = time_point_cast<nanoseconds>(system_clock::now());
			int randSeed = now.time_since_epoch().count();
			// broadcast the same root seed to all processors
			ParallelDescriptor::Bcast(&randSeed,1,ParallelDescriptor::IOProcessorNumber());
			InitRandom(randSeed+ParallelDescriptor::MyProc());
		} else {
			Abort("Must supply non-negative seed");
		}

		ba.define(domain);
		ba.maxSize(IntVect(max_grid_size));
		dmap.define(ba);
		
		/*
		0	- n - number density
		1	- rho - mass density
		2	- Jx - x-mom density
		3	- Jy - y-mom density
		4	- Jz - z-mom density
		5	- tau_xx - xx shear stress
		6	- tau_xy - xy shear stress
		7	- tau_xz - xz shear stress
		8	- tau_yy - yy shear stress
		9 - tau_yz - yz shear stress
		10 - tau_zz - zz shear stress
		11 - E - energy
		12 - T_gran - granular temperature
		13 - X_i - mole fraction for spec. i
		14 - Y_i - mass fraction for spec. i
		15 - u_i - x-vel for spec. i
		16 - v_i - y-vel for spec. i
		17 - w_i - z-vel for spec. i
		18 - uu_i 
		19 - uv_i
		20 - uw_i
		21 - vv_i
		22 - vw_i
		23 - ww_i
		24 - E_i
		25 - T_g_i
		... (repeat for each add. species)
		*/
		nstats = (nspecies+1)*13;
		cu.define(ba, dmap, nstats, 0); cu.setVal(0.);
		cuMeans.define(ba, dmap, nstats, 0); cuMeans.setVal(0.);
		cuVars.define(ba, dmap, nstats, 0);  cuVars.setVal(0.);
		
		// Variances
		// Each has (nspecies)(nspecies+1)*0.5 data points
		/*
		0  - drho.dJx
		1  - drho.dJy
		2  - drho.dJz
		3  - drho.dE
		4  - dJx.dJy
		5  - dJx.dJz
		6  - dJy.dJz
		7 - dJx.dE
		8 - dJy.dE
		9 - dJz.dE
    // Comment -- need to add some temperature-velocity covariances as well (see Balakrishnan)
		*/
		// Add multifabs for variances
		//nnspec = std::ceil((double)nspecies*(nspecies-1)*0.5);
		//MultiFab coVars(ba, dmap, (nnspec+1)*15, 0);
		coVars.define(ba, dmap, 10, 0);
		// just track ones you want
	} else {
		// restart from checkpoint
	}

	Vector<int> is_periodic (AMREX_SPACEDIM,0);
	for (int i=0; i<AMREX_SPACEDIM; ++i) {
		if (bc_vel_lo[i] == -1 && bc_vel_hi[i] == -1) {
			is_periodic [i] = -1;
		}
	}

	// This defines a Geometry object
	RealBox realDomain({AMREX_D_DECL(prob_lo[0],prob_lo[1],prob_lo[2])},
		{AMREX_D_DECL(prob_hi[0],prob_hi[1],prob_hi[2])});

	Geometry geom (domain ,&realDomain,CoordSys::cartesian,is_periodic.data());
	const Real* dx = geom.CellSize();

	// Currently overwritten later
	Real dt = fixed_dt;

	int paramPlaneCount = 6;
	paramPlane paramPlaneList[paramPlaneCount];
	BuildParamplanes(paramPlaneList,paramPlaneCount,realDomain.lo(),realDomain.hi());

	// Particle tile size
	Vector<int> ts(BL_SPACEDIM);
    
	for (int d=0; d<AMREX_SPACEDIM; ++d) {        
		if (max_particle_tile_size[d] > 0) {
			ts[d] = max_particle_tile_size[d];
		}
		else {
			ts[d] = max_grid_size[d];
		}
	}

	ParmParse pp ("particles");
	pp.addarr("tile_size", ts);

	int cRange = 0;
	FhdParticleContainer particles(geom, dmap, ba, cRange);
	if (restart < 0 && particle_restart < 0) {
		// Collision Cell Vars
		particles.mfselect.define(ba, dmap, nspecies*nspecies, 0);
		particles.mfselect.setVal(0.);
		
		particles.mfphi.define(ba, dmap, nspecies, 0);
		particles.mfphi.setVal(0.);
		
		particles.mfvrmax.define(ba, dmap, nspecies*nspecies, 0);
		particles.mfvrmax.setVal(0.);
		// overwrite dt
		particles.InitParticles(dt);

		particles.InitCollisionCells();
		amrex::Print() << "Overwritten dt so Courant number <1: " << dt << "\n";
	}
	
	else {

	}

	Real init_time = ParallelDescriptor::second() - strt_time;
	ParallelDescriptor::ReduceRealMax(init_time);
	amrex::Print() << "Initialization time = " << init_time << " seconds " << std::endl;

	// Frequency of plot and checkpoints
	int plot_step = 10;
	//int check_step = 10;
	Real time = 0.;
	int statsCount = 1;
	int plotCount = 0;
	int step_stat = 1;
	for (int istep=0; istep<=max_step; ++istep) {
		Real tbegin = ParallelDescriptor::second();
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Checkpoints
		//if (plot_int > 0 && istep > 0 && istep%plot_step == 0) {
           
		//}

		//if (istep==1 && restart > 0) {
      //     ReadCheckPoint(istep, statsCount, time, geom,
      //     						cu, cuMeans, cuVars, coVars);//,
           //						particles.mfselect, particles.mfphi, particles.mfvrmax);
		//}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Collide Particles
		//amrex::Print() << "Collisions per particles per species: " << 
		//	particles.CountedCollision[
		particles.CalcSelections(dt);
		particles.CollideParticles(dt);
		
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Move Particles
		particles.MoveParticlesCPP(dt, paramPlaneList, paramPlaneCount);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Stats
		if(istep%n_steps_skip == 0) {cu.setVal(0.); particles.EvaluateStats(cu, cuMeans, cuVars, coVars, statsCount++);}
		if(istep%plot_int == 0) {particles.writePlotFile(cu, cuMeans, cuVars,  coVars,  geom,  time, plotCount++);}
 
		Real tend = ParallelDescriptor::second() - tbegin;
		ParallelDescriptor::ReduceRealMax(tend);
		amrex::Print() << "Advanced step " << istep << " in " << tend << " seconds\n";
	}

	Real stop_time = ParallelDescriptor::second() - strt_time;
	ParallelDescriptor::ReduceRealMax(stop_time);
	amrex::Print() << "Run time = " << stop_time << " seconds" << std::endl;


	//if(particle_input<0 && ParallelDescriptor::MyProc() == 0) {particles.OutputParticles();} // initial condition
}