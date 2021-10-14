#include "common_functions.H"


#include <AMReX_VisMF.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Vector.H>

#include "StructFact.H"

using namespace amrex;

void WritePlotFilesSF_2D(const amrex::MultiFab& mag, const amrex::MultiFab& realimag, const amrex::Geometry& geom,
                         const int step, const Real time, const amrex::Vector< std::string >& names, std::string plotfile_base) {

      // Magnitude of the Structure Factor
      std::string name = plotfile_base;
      name += "_mag";
      const std::string plotfilename1 = amrex::Concatenate(name,step,9);
      
      Real dx0 = geom.CellSize(0);
      Real dx1 = geom.CellSize(1);
      Real dx2 = geom.CellSize(2);
      Real pi = 3.1415926535897932;
      Box domain = geom.Domain();
      RealBox real_box({AMREX_D_DECL(-pi/dx0,-pi/dx1,-pi/dx2)},
                       {AMREX_D_DECL( pi/dx0, pi/dx1, pi/dx2)});
      Vector<int> is_periodic(AMREX_SPACEDIM,0);  // set to 0 (not periodic) by default
      for (int i=0; i<AMREX_SPACEDIM; ++i) {
          is_periodic[i] = geom.isPeriodic(i);
      }
      Geometry geom2;
      geom2.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());

      Vector<std::string> varNames;
      varNames.resize(names.size());
      for (int n=0; n<names.size(); n++) {
          varNames[n] = names[n];
      }

      WriteSingleLevelPlotfile(plotfilename1,mag,varNames,geom2,time,step);

      // Components of the Structure Factor
      name = plotfile_base;
      name += "_real_imag";
      const std::string plotfilename2 = amrex::Concatenate(name,step,9);

      varNames.resize(2*names.size());
      int cnt = 0; // keep a counter for plotfile variables
      for (int n=0; n<names.size(); n++) {
          varNames[cnt] = names[cnt];
          varNames[cnt] += "_real";
          cnt++;
      }

      int index = 0;
      for (int n=0; n<names.size(); n++) {
          varNames[cnt] = names[index];
          varNames[cnt] += "_imag";
          index++;
          cnt++;
      }

      WriteSingleLevelPlotfile(plotfilename2,realimag,varNames,geom2,time,step);
}

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

    // this brace is used for some temporaries used to build geom
    // so everything in here goes out of scope
    // this section is used to build the geometry object for flattened
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

    /**************************************
    // 2D averaged structure factor testing
    ***************************************/

    if (project_dir != 2) {
        Abort("supply project_dir = 2 for the 2D average test");
    }
      
    Vector < StructFact > structFact_2D_array;
    structFact_2D_array.resize(n_cells[2]);
    
    Geometry geom_flat_2D;

    MultiFab mf_flat_temp; // temporary MF for getting the correct geom_flat_2D and ba_flat_2D
    mf_flat_temp.define(ba, dmap, 2, 0);
    mf_flat_temp.setVal(0.0);
    ComputeVerticalAverage(mf_full, mf_flat_temp, geom, 2, 0, 2);
    MultiFab mf_flat_temp_rot = RotateFlattenedMF(mf_flat_temp);

    BoxArray ba_flat_2D = mf_flat_temp_rot.boxArray();
    const DistributionMapping& dmap_flat_2D = mf_flat_temp_rot.DistributionMap();
    {
        IntVect dom_lo(AMREX_D_DECL(0,0,0));
        IntVect dom_hi;
        dom_hi[0] = n_cells[0]-1;
        dom_hi[1] = n_cells[1]-1;
        dom_hi[2] = 0;

        Box domain(dom_lo, dom_hi);

        // This defines the physical box
        Vector<Real> projected_hi(AMREX_SPACEDIM);

        // yes you could simplify this code but for now
        // these are written out fully to better understand what is happening
        // we wanted projected_hi[AMREX_SPACEDIM-1] to be equal to dx[projected_dir]
        // and need to transmute the other indices depending on project_dir
        projected_hi[0] = prob_hi[0];
        projected_hi[1] = prob_hi[1];
        projected_hi[2] = prob_hi[project_dir] / n_cells[project_dir];

        RealBox real_box({AMREX_D_DECL(     prob_lo[0],     prob_lo[1],     prob_lo[2])},
                         {AMREX_D_DECL(projected_hi[0],projected_hi[1],projected_hi[2])});
        
        // This defines a Geometry object
        geom_flat_2D.define(domain,&real_box,CoordSys::cartesian,is_periodic.data());
    }

    for (int i = 0; i < n_cells[2]; ++i) { 
        structFact_2D_array[i].define(ba_flat_2D,dmap_flat_2D,var_names,var_scaling);
    }

    for (int i=0; i<n_cells[2]; ++i) {
       MultiFab mf_flat_2D, mf_flat_2D_rot;
       ExtractSliceI(mf_full, mf_flat_2D, geom, 2, i, 0, 2);
       mf_flat_2D_rot = RotateFlattenedMF(mf_flat_2D);
       structFact_2D_array[i].FortStructure(mf_flat_2D_rot,geom_flat_2D);
    }

    MultiFab mag_2D;
    MultiFab realimag_2D;
    mag_2D.define(ba_flat_2D,dmap_flat_2D,structFact_2D_array[0].get_ncov(),0);
    realimag_2D.define(ba_flat_2D,dmap_flat_2D,2*structFact_2D_array[0].get_ncov(),0);

    mag_2D.setVal(0.0);
    realimag_2D.setVal(0.0);

    for (int i=0; i<n_cells[2]; ++i) {
        structFact_2D_array[i].AddToExternal(mag_2D,realimag_2D,geom_flat_2D);
    }
    
    Real ncellsinv = 1.0/n_cells[2];
    mag_2D.mult(ncellsinv);
    realimag_2D.mult(ncellsinv);

    WritePlotFilesSF_2D(mag_2D, realimag_2D, geom_flat_2D, 0, 0.0, structFact_2D_array[0].get_names(), "plt_SF_flat_2D");

    // Call the timer again and compute the maximum difference between the start time
    // and stop time over all processors
    Real stop_time = ParallelDescriptor::second() - strt_time;
    ParallelDescriptor::ReduceRealMax(stop_time);
    amrex::Print() << "Run time = " << stop_time << std::endl;

}
