
#include "common_functions.H"
#include "hydro_functions_F.H"
#include "StructFact.H"

#include "AMReX_PlotFileUtil.H"

StructFact::StructFact(BoxArray ba_in, DistributionMapping dmap_in) {

  // static_assert(AMREX_SPACEDIM == 3, "3D only");

  // Note that we are defining with NO ghost cells

  umac_dft_real.resize(AMREX_SPACEDIM);
  umac_dft_imag.resize(AMREX_SPACEDIM);
  cov_real.resize(COV_NVAR);
  cov_imag.resize(COV_NVAR);
  struct_umac.resize(COV_NVAR);
  
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    umac_dft_real[d].define(ba_in, dmap_in, 1, 0);
    umac_dft_imag[d].define(ba_in, dmap_in, 1, 0);
  }

  for (int d=0; d<COV_NVAR; d++) {
    cov_real[d].define(ba_in, dmap_in, 1, 0);
    cov_imag[d].define(ba_in, dmap_in, 1, 0);
    cov_real[d].setVal(0.0);
    cov_imag[d].setVal(0.0);
  }

  for (int d=0; d<COV_NVAR; d++) {
    struct_umac[d].define(ba_in, dmap_in, 1, 0);
    struct_umac[d].setVal(0.0);
  }
}

void StructFact::FortStructure(const std::array< MultiFab, AMREX_SPACEDIM >& umac, Geometry geom) {

  const BoxArray& ba = umac_dft_real[0].boxArray();
  // const BoxArray& ba = amrex::enclosedCells(ba_nodal);
  const DistributionMapping& dm = umac[0].DistributionMap();

  if (ba.size() != 1) {
    Abort("Only adapted for single grid");
    exit(0);
  }
  
  std::array< MultiFab, AMREX_SPACEDIM > umac_cc;
  for (int d=0; d<AMREX_SPACEDIM; d++) {
    umac_cc[d].define(ba, dm, 1, 0);
  }

  MultiFab cov_real_temp;
  MultiFab cov_imag_temp;
  cov_real_temp.define(ba, dm, 1, 0);
  cov_imag_temp.define(ba, dm, 1, 0);
  
  for (int d=0; d<AMREX_SPACEDIM; d++) { 
    AverageFaceToCC(umac[d], 0, umac_cc[d], 0, 1);
  }

  ComputeFFT(umac_cc, geom);
  
  // Note: Covariances are contained in this vector, as opposed to correlations, which are normalized
  int index = 0;
  for (int j=0; j<AMREX_SPACEDIM; j++) {
    for (int i=j; i<AMREX_SPACEDIM; i++) {

      // Compute temporary real and imaginary components of covariance
      cov_real_temp.setVal(0.0);
      MultiFab::AddProduct(cov_real_temp,umac_dft_real[i],0,umac_dft_real[j],0,0,1,0);
      MultiFab::AddProduct(cov_real_temp,umac_dft_imag[i],0,umac_dft_imag[j],0,0,1,0);
      cov_imag_temp.setVal(0.0);
      MultiFab::AddProduct(cov_imag_temp,umac_dft_real[i],0,umac_dft_imag[j],0,0,1,0);
      cov_imag_temp.mult(-1.0,0);
      MultiFab::AddProduct(cov_imag_temp,umac_dft_imag[i],0,umac_dft_real[j],0,0,1,0);
      
      MultiFab::Add(cov_real[index],cov_real_temp,0,0,1,0);
      MultiFab::Add(cov_imag[index],cov_imag_temp,0,0,1,0);
      
      index++;
    }
  }

  bool write_data = false;
  if (write_data) {
    index = 0;
    std::string plotname = "a_COV"; 
    std::string x;
    for (int j=0; j<AMREX_SPACEDIM; j++) {
      for (int i=j; i<AMREX_SPACEDIM; i++) {
	x = plotname;
	x += '_';
	x += '0' + i;
	x += '0' + j;
	VisMF::Write(cov_real[index],x);
	index++;
      }
    }
  }

  nsamples++;
}

void StructFact::WritePlotFile(const int step, const amrex::Real time, const amrex::Geometry geom,
			       const amrex::Real scale) {
  
  amrex::Vector< MultiFab > struct_temp;
  struct_temp.resize(COV_NVAR);
  for (int d=0; d<COV_NVAR; d++) {
    struct_temp[d].define(struct_umac[0].boxArray(), struct_umac[0].DistributionMap(), 1, 0);
    struct_temp[d].setVal(0.0);
  }

  StructFinalize(scale);
  for(int d=0; d<COV_NVAR; d++) {   
    MultiFab::Copy(struct_temp[d],struct_umac[d],0,0,1,0);
  }

  // Print() << "number of samples = " << nsamples << std::endl;

  ShiftFFT(struct_temp);

  // Write out structure factor to plot file
  const std::string plotfilename = "plt_structure_factor";
  int nPlot = COV_NVAR;
  MultiFab plotfile(struct_umac[0].boxArray(), struct_umac[0].DistributionMap(), nPlot, 0);
  Vector<std::string> varNames(nPlot);

  // keep a counter for plotfile variables
  int cnt = 0;

  std::string plotname = "structFact";
  std::string x;
  for (int j=0; j<AMREX_SPACEDIM; j++) {
    for (int i=j; i<AMREX_SPACEDIM; i++) {
      x = plotname;
      x += '_';
      x += (120+j);
      x += (120+i);
      varNames[cnt++] = x;
    }
  }

  // reset plotfile variable counter
  cnt = 0;

  // copy structure factor into plotfile
  for (int d=0; d<COV_NVAR; ++d) {
    MultiFab::Copy(plotfile, struct_temp[cnt], 0, cnt, 1, 0);
    cnt++;
  }

  // write a plotfile
  WriteSingleLevelPlotfile(plotfilename,plotfile,varNames,geom,time,step);
}

void StructFact::StructOut(amrex::Vector< MultiFab >& struct_out, const amrex::Real scale) {
  
  struct_out.resize(COV_NVAR);
  for (int d=0; d<COV_NVAR; d++) {
    struct_out[d].define(struct_umac[0].boxArray(), struct_umac[0].DistributionMap(), 1, 0);
    struct_out[d].setVal(0.0);
  }

  StructFinalize(scale);
  for(int d=0; d<COV_NVAR; d++) {   
    MultiFab::Copy(struct_out[d],struct_umac[d],0,0,1,0);
  }

  ShiftFFT(struct_out);
}

void StructFact::StructFinalize(const amrex::Real scale) {
  Real nsamples_inv = 1.0/(double)nsamples;

  for(int d=0; d<COV_NVAR; d++) {
    MultiFab::AddProduct(struct_umac[d],cov_real[d],0,cov_real[d],0,0,1,0);
    MultiFab::AddProduct(struct_umac[d],cov_imag[d],0,cov_imag[d],0,0,1,0);
  }

  SqrtMF(struct_umac);

  for(int d=0; d<COV_NVAR; d++) {
    struct_umac[d].mult(nsamples_inv,0);
    struct_umac[d].mult(scale,0);
  }
}

void StructFact::ComputeFFT(const std::array< MultiFab, AMREX_SPACEDIM >& umac_cc, Geometry geom) {

  const BoxArray& ba = umac_dft_real[0].boxArray();
  // amrex::Print() << "BA " << ba << std::endl;

  if (umac_dft_real[0].nGrow() != 0 || umac_dft_real[0].nGrow() != 0) 
    amrex::Error("Current implementation requires that both umac_cc[0] and umac_dft_real[0] have no ghost cells");

  // We assume that all grids have the same size hence 
  // we have the same nx,ny,nz on all ranks
  int nx = ba[0].size()[0];
  int ny = ba[0].size()[1];
#if (AMREX_SPACEDIM == 2)
  int nz = 1;
#elif (AMREX_SPACEDIM == 3)
  int nz = ba[0].size()[2];
#endif

  Box domain(geom.Domain());

  int nbx = domain.length(0) / nx;
  int nby = domain.length(1) / ny;
#if (AMREX_SPACEDIM == 2)
  int nbz = 1;
#elif (AMREX_SPACEDIM == 3)
  int nbz = domain.length(2) / nz;
#endif
  int nboxes = nbx * nby * nbz;
  if (nboxes != ba.size()) 
    amrex::Error("NBOXES NOT COMPUTED CORRECTLY");
  // amrex::Print() << "Number of boxes:\t" << nboxes << std::endl;

  Vector<int> rank_mapping;
  rank_mapping.resize(nboxes);

  DistributionMapping dmap = umac_cc[0].DistributionMap();

  for (int ib = 0; ib < nboxes; ++ib)
    {
      int i = ba[ib].smallEnd(0) / nx;
      int j = ba[ib].smallEnd(1) / ny;
#if (AMREX_SPACEDIM == 2)
      int k = 0;
#elif (AMREX_SPACEDIM == 3)
      int k = ba[ib].smallEnd(2) / nz;
#endif

      // This would be the "correct" local index if the data wasn't being transformed
      // int local_index = k*nbx*nby + j*nbx + i;

      // This is what we pass to dfft to compensate for the Fortran ordering
      //      of amrex data in MultiFabs.
      int local_index = i*nby*nbz + j*nbz + k;

      rank_mapping[local_index] = dmap[ib];
      // if (verbose)
      // 	amrex::Print() << "LOADING RANK NUMBER " << dmap[ib] << " FOR GRID NUMBER " << ib 
      // 		       << " WHICH IS LOCAL NUMBER " << local_index << std::endl;
    }

  // FIXME: Assumes same grid spacing
  // Real h = geom.CellSize(0);
  // Real hsq = h*h;

  // Assume for now that nx = ny = nz
#if (AMREX_SPACEDIM == 2)
  int Ndims[3] = { nbz, nby};
  int     n[3] = {domain.length(2), domain.length(1)};
#elif (AMREX_SPACEDIM == 3)
  int Ndims[3] = { nbz, nby, nbx };
  int     n[3] = {domain.length(2), domain.length(1), domain.length(0)};
#endif
  hacc::Distribution d(MPI_COMM_WORLD,n,Ndims,&rank_mapping[0]);
  hacc::Dfft dfft(d);

  for (int dim=0; dim<AMREX_SPACEDIM; dim++) {
    for (MFIter mfi(umac_cc[0],false); mfi.isValid(); ++mfi)
      {
	// int gid = mfi.index();
	// size_t local_size  = dfft.local_size();
   
	std::vector<complex_t, hacc::AlignedAllocator<complex_t, ALIGN> > a;
	std::vector<complex_t, hacc::AlignedAllocator<complex_t, ALIGN> > b;

	a.resize(nx*ny*nz);
	b.resize(nx*ny*nz);

	dfft.makePlans(&a[0],&b[0],&a[0],&b[0]);

	// *******************************************
	// Copy real data from Rhs into real part of a -- no ghost cells and
	// put into C++ ordering (not Fortran)
	// *******************************************
	complex_t zero(0.0, 0.0);
	size_t local_indx = 0;
	for(size_t k=0; k<(size_t)nz; k++) {
	  for(size_t j=0; j<(size_t)ny; j++) {
	    for(size_t i=0; i<(size_t)nx; i++) {

	      complex_t temp(umac_cc[dim][mfi].dataPtr()[local_indx],0.);
	      a[local_indx] = temp;
	      local_indx++;

	    }
	  }
	}

	//  *******************************************
	//  Compute the forward transform
	//  *******************************************
	dfft.forward(&a[0]);
	
	// Note: Scaling for inverse FFT
	size_t global_size  = dfft.global_size();
	
	Real pi = 4.0*std::atan(1.0);
	double fac = sqrt(1.0 / global_size / pi);

	local_indx = 0;
	for(size_t k=0; k<(size_t)nz; k++) {
	  for(size_t j=0; j<(size_t)ny; j++) {
	    for(size_t i=0; i<(size_t)nx; i++) {

	      // Divide by 2 pi N
	      umac_dft_real[dim][mfi].dataPtr()[local_indx] = fac * std::real(a[local_indx]);
	      umac_dft_imag[dim][mfi].dataPtr()[local_indx] = fac * std::imag(a[local_indx]);
	      local_indx++;
	    }
	  }
	}
      }
  }

  bool write_data = false;
  if (write_data) {
    std::string plotname = "a_DFT"; 
    std::string x;
    for (int i=0; i<AMREX_SPACEDIM; i++) {
      x = plotname;
      x += '_';
      x += '0' + i;
      VisMF::Write(umac_dft_real[i],x);
    }
  }
}

void StructFact::ShiftFFT(amrex::Vector< MultiFab >& struct_out) {
  // NOT PARALLELIZED
  // Shift DFT by N/2+1 (pi)
  for(int d=0; d<COV_NVAR; d++) {
    for (MFIter mfi(struct_out[0]); mfi.isValid(); ++mfi) {
      // Note: Make sure that multifab is cell-centered
      const Box& validBox = mfi.validbox();
      fft_shift(ARLIM_3D(validBox.loVect()), ARLIM_3D(validBox.hiVect()),
  		BL_TO_FORTRAN_ANYD(struct_out[d][mfi]));
    }
  }
}

void StructFact::SqrtMF(amrex::Vector< MultiFab >& struct_out) {
  for(int d=0; d<COV_NVAR; d++) {
    for (MFIter mfi(struct_out[0]); mfi.isValid(); ++mfi) {
      // Note: Make sure that multifab is cell-centered
      const Box& validBox = mfi.validbox();
      sqrt_mf(ARLIM_3D(validBox.loVect()), ARLIM_3D(validBox.hiVect()),
	      BL_TO_FORTRAN_ANYD(struct_out[d][mfi]));
    }
  }
}
