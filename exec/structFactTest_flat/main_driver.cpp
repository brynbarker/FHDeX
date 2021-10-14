#include "common_functions.H"


#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>

#include "StructFact.H"

using namespace amrex;

// argv contains the name of the inputs file entered at the command line
void main_driver(const char* argv)
{

    BL_PROFILE_VAR("main_driver()",main_driver);

    // store the current time so we can later compute total run time.
    Real strt_time = ParallelDescriptor::second();

    std::string inputs_file = argv;

    // copy contents of F90 modules to C++ namespaces
    InitializeCommonNamespace();

    // is the problem periodic?
    Vector<int> is_periodic(AMREX_SPACEDIM,1);  // set to 1 (periodic) by default

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

    const Real* dx = geom.CellSize();
    GpuArray<Real,AMREX_SPACEDIM> dx_gpu{AMREX_D_DECL(dx[0], dx[1], dx[2])};

    // domain length
    RealVect L(AMREX_D_DECL(prob_hi[0]-prob_lo[0],
                            prob_hi[1]-prob_lo[1],
                            prob_hi[2]-prob_lo[2]));

    Real pi = 3.141592653589793;

    // use dVol for scaling
    Real dVol = (AMREX_SPACEDIM == 2) ? dx[0]*dx[1]*cell_depth : dx[0]*dx[1]*dx[2];
    
    // how boxes are distrubuted among MPI processes
    DistributionMapping dmap(ba);
    
    /////////////////////////////////////////

    Vector< std::string > var_names(2);
    var_names[0] = "phi1";
    var_names[1] = "phi2";

    // for the 3 pairs
    Vector< Real > var_scaling(3,1./dVol);
    
    MultiFab mf_full;
    mf_full.define(ba, dmap, 2, 0);

    // WRITE INIT ROUTINE
    mf_full.setVal(0.);

    for (MFIter mfi(mf_full,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        const Array4<Real>& struct_fab = mf_full.array(mfi);

        amrex::ParallelFor(bx, 2, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real x = prob_lo[0] + (i+0.5)*dx_gpu[0];
            Real y = prob_lo[1] + (j+0.5)*dx_gpu[1];
            Real z = prob_lo[2] + (k+0.5)*dx_gpu[2];

            if (project_dir == 0) {
                struct_fab(i,j,k,n) = sin(4.*pi*y / L[1]) + sin(4.*pi*z / L[2]) + sin(4.*pi*y / L[1]) * sin(4.*pi*z / L[2]);

            } else if (project_dir == 1) {
                struct_fab(i,j,k,n) = sin(4.*pi*x / L[0]) + sin(4.*pi*z / L[2]) + sin(4.*pi*x / L[0]) * sin(4.*pi*z / L[2]);

            } else if (project_dir == 2) {
                struct_fab(i,j,k,n) = sin(4.*pi*x / L[0]) + sin(4.*pi*y / L[1]) + sin(4.*pi*x / L[0]) * sin(4.*pi*y / L[1]);
            } else {
                Abort("Invalid project_dir");
            }
                
        });

    }
    
    amrex::Vector< int > s_pairA(3);
    amrex::Vector< int > s_pairB(3);

    // Select which variable pairs to include in structure factor:
    s_pairA[0] = 0;
    s_pairB[0] = 0;
    //
    s_pairA[1] = 0;
    s_pairB[1] = 1;
    //
    s_pairA[2] = 1;
    s_pairB[2] = 1;

    /**************************************
    // full multifab structure factor testing
    ***************************************/
    
    StructFact structFact(ba,dmap,var_names,var_scaling,s_pairA,s_pairB);
    
    structFact.FortStructure(mf_full,geom);
      
    structFact.WritePlotFile(0,0.,geom,"plt_SF");

    /**************************************
    // flattened structure factor testing
    ***************************************/

    if (project_dir < 0) {
        Abort("supply project_dir >= 0 for this test");
    }
    
    StructFact structFact_flat;
    
    Geometry geom_flat;
    MultiFab mf_flat;

    if (slicepoint < 0) {
        // compute vertical average over project_dir
        ComputeVerticalAverage(mf_full, mf_flat, geom, project_dir, 0, 2);
    } else {
        // extract a slice at slicepoint in the project_dir plane
        ExtractSlice(mf_full, mf_flat, geom, project_dir, 0, 2);        
    }
    
    // we rotate this flattened MultiFab to have normal in the z-direction since
    // our structure factor class assumes this for flattened
    MultiFab mf_flat_rot = RotateFlattenedMF(mf_flat);

    BoxArray ba_flat = mf_flat_rot.boxArray();
    const DistributionMapping& dmap_flat = mf_flat_rot.DistributionMap();

    {
        IntVect dom_lo(AMREX_D_DECL(0,0,0));
        IntVect dom_hi;

        // yes you could simplify this code but for now
        // these are written out fully to better understand what is happening
        // we wanted dom_hi[AMREX_SPACEDIM-1] to be equal to 0
        // and need to transmute the other indices depending on project_dir
#if (AMREX_SPACEDIM == 2)
        if (project_dir == 0) {
            dom_hi[0] = n_cells[1]-1;
        }
        else if (project_dir == 1) {
            dom_hi[0] = n_cells[0]-1;
        }
        dom_hi[1] = 0;
#elif (AMREX_SPACEDIM == 3)
        if (project_dir == 0) {
            dom_hi[0] = n_cells[1]-1;
            dom_hi[1] = n_cells[2]-1;
        } else if (project_dir == 1) {
            dom_hi[0] = n_cells[0]-1;
            dom_hi[1] = n_cells[2]-1;
        } else if (project_dir == 2) {
            dom_hi[0] = n_cells[0]-1;
            dom_hi[1] = n_cells[1]-1;
        }
        dom_hi[2] = 0;
#endif
        Box domain(dom_lo, dom_hi);

        // This defines the physical box
        Vector<Real> projected_hi(AMREX_SPACEDIM);

        // yes you could simplify this code but for now
        // these are written out fully to better understand what is happening
        // we wanted projected_hi[AMREX_SPACEDIM-1] to be equal to dx[projected_dir]
        // and need to transmute the other indices depending on project_dir
#if (AMREX_SPACEDIM == 2)
        if (project_dir == 0) {
            projected_hi[0] = prob_hi[1];
        } else if (project_dir == 1) {
            projected_hi[0] = prob_hi[0];
        }
        projected_hi[1] = prob_hi[project_dir] / n_cells[project_dir];
#elif (AMREX_SPACEDIM == 3)
        if (project_dir == 0) {
            projected_hi[0] = prob_hi[1];
            projected_hi[1] = prob_hi[2];
        } else if (project_dir == 1) {
            projected_hi[0] = prob_hi[0];
            projected_hi[1] = prob_hi[2];
        } else if (project_dir == 2) {
            projected_hi[0] = prob_hi[0];
            projected_hi[1] = prob_hi[1];
        }
        projected_hi[2] = prob_hi[project_dir] / n_cells[project_dir];
#endif

        RealBox real_box({AMREX_D_DECL(     prob_lo[0],     prob_lo[1],     prob_lo[2])},
                         {AMREX_D_DECL(projected_hi[0],projected_hi[1],projected_hi[2])});
        
        // This defines a Geometry object
        geom_flat.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());
    }

    structFact_flat.define(ba_flat,dmap_flat,var_names,var_scaling);

    structFact_flat.FortStructure(mf_flat_rot,geom_flat);

    structFact_flat.WritePlotFile(0,0.,geom_flat,"plt_SF_flat");    
      
    // Call the timer again and compute the maximum difference between the start time
    // and stop time over all processors
    Real stop_time = ParallelDescriptor::second() - strt_time;
    ParallelDescriptor::ReduceRealMax(stop_time);
    amrex::Print() << "Run time = " << stop_time << std::endl;

}
