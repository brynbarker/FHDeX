#include "compressible_functions.H"
#include "common_functions.H"

void calculateFlux(const MultiFab& cons_in, const MultiFab& prim_in,
                   const MultiFab& eta_in, const MultiFab& zeta_in, const MultiFab& kappa_in,
                   const MultiFab& chi_in, const MultiFab& D_in,
                   std::array<MultiFab, AMREX_SPACEDIM>& flux_in,
                   std::array<MultiFab, AMREX_SPACEDIM>& stochFlux_in,
                   std::array<MultiFab, AMREX_SPACEDIM>& cornx_in,
                   std::array<MultiFab, AMREX_SPACEDIM>& corny_in,
                   std::array<MultiFab, AMREX_SPACEDIM>& cornz_in,
                   MultiFab& visccorn_in,
                   MultiFab& rancorn_in,
                   const amrex::Geometry geom,
		   const amrex::Vector< amrex::Real >& stoch_weights,
		   const amrex::Real* dx, const amrex::Real dt)
{
    BL_PROFILE_VAR("calculateFlux_In()",calculateFlux_In);
    
    AMREX_D_TERM(flux_in[0].setVal(0);,
                 flux_in[1].setVal(0);,
                 flux_in[2].setVal(0););

    // Loop over boxes
    for ( MFIter mfi(cons_in); mfi.isValid(); ++mfi) {
        
        const Box& bx = mfi.tilebox();

        //NOTE: Must do stoch. flux_ines first, 
	//      because flux_ines at boundaries are weighted according to BCs
        if(stoch_stress_form == 1)
        { 
            stoch_flux(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
                       cons_in[mfi].dataPtr(),  
                       prim_in[mfi].dataPtr(),    
                       flux_in[0][mfi].dataPtr(),
                       flux_in[1][mfi].dataPtr(),
#if (AMREX_SPACEDIM == 3)
                       flux_in[2][mfi].dataPtr(),
#endif
                       stochFlux_in[0][mfi].dataPtr(),
                       stochFlux_in[1][mfi].dataPtr(),
#if (AMREX_SPACEDIM == 3)
                       stochFlux_in[2][mfi].dataPtr(),
#endif
                       rancorn_in[mfi].dataPtr(),
                       eta_in[mfi].dataPtr(),  
                       zeta_in[mfi].dataPtr(),  
                       kappa_in[mfi].dataPtr(),
                       chi_in[mfi].dataPtr(),  
                       D_in[mfi].dataPtr(),  
                       ZFILL(dx), &dt);
	}
    }

    
    // Loop over boxes
    for ( MFIter mfi(cons_in); mfi.isValid(); ++mfi) {

#if 1

        AMREX_D_TERM(const Array4<Real>& fluxx = flux_in[0].array(mfi); ,
                     const Array4<Real>& fluxy = flux_in[1].array(mfi); ,
                     const Array4<Real>& fluxz = flux_in[2].array(mfi));

        const Array4<const Real> prim = prim_in.array(mfi);
        const Array4<const Real> cons = cons_in.array(mfi);
        
        const Array4<const Real> eta   = eta_in.array(mfi);
        const Array4<const Real> zeta  = zeta_in.array(mfi);
        const Array4<const Real> kappa = kappa_in.array(mfi);
        const Array4<const Real> chi   = chi_in.array(mfi);
        const Array4<const Real> Dij   = D_in.array(mfi);

        const Array4<Real> cornux = cornx_in[0].array(mfi);
        const Array4<Real> cornvx = cornx_in[1].array(mfi);
        const Array4<Real> cornwx = cornx_in[2].array(mfi);
        const Array4<Real> cornuy = corny_in[0].array(mfi);
        const Array4<Real> cornvy = corny_in[1].array(mfi);
        const Array4<Real> cornwy = corny_in[2].array(mfi);
        const Array4<Real> cornuz = cornz_in[0].array(mfi);
        const Array4<Real> cornvz = cornz_in[1].array(mfi);
        const Array4<Real> cornwz = cornz_in[2].array(mfi);
        const Array4<Real> visccorn = visccorn_in.array(mfi);
        
        const Box& tbx = mfi.nodaltilebox(0);
        const Box& tby = mfi.nodaltilebox(1);
        const Box& tbz = mfi.nodaltilebox(2);

        IntVect nd(AMREX_D_DECL(1,1,1));
        const Box& tbn = mfi.tilebox(nd);

        Real dxinv = 1./dx[0];
        Real half = 0.5;
        Real two = 2.;
        Real twothirds = 2./3.;
        Real onetwelfth = 1./12.;
        
        // from namelist
        int nspecies_gpu = nspecies;
        int algorithm_type_gpu = algorithm_type;
        int visc_type_gpu = visc_type;
        int n_cells_z = n_cells[2];
    
        GpuArray<Real,MAX_SPECIES> hcp_gpu;
        for (int n=0; n<nspecies; ++n) {
            hcp_gpu[n] = hcp[n];
        }    
        
        amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                GpuArray<Real,MAX_SPECIES> meanXk;
                GpuArray<Real,MAX_SPECIES> meanYk;
                GpuArray<Real,MAX_SPECIES> dk;
                GpuArray<Real,MAX_SPECIES> Fk;
                GpuArray<Real,MAX_SPECIES> hk;
                GpuArray<Real,MAX_SPECIES> soret;

                Real muxp = half*(eta(i,j,k) + eta(i-1,j,k));
                Real kxp = half*(kappa(i,j,k) + kappa(i-1,j,k));

                Real tauxxp = muxp*(prim(i,j,k,1) - prim(i-1,j,k,1))/dx[0];
                Real tauyxp = muxp*(prim(i,j,k,2) - prim(i-1,j,k,2))/dx[0];
                Real tauzxp = muxp*(prim(i,j,k,3) - prim(i-1,j,k,3))/dx[0];

                Real divxp = 0;

                Real phiflx =  tauxxp*(prim(i-1,j,k,1)+prim(i,j,k,1))
                    +  divxp*(prim(i-1,j,k,1)+prim(i,j,k,1))
                    +  tauyxp*(prim(i-1,j,k,2)+prim(i,j,k,2))
                    +  tauzxp*(prim(i-1,j,k,3)+prim(i,j,k,3));
           
                fluxx(i,j,k,1) = fluxx(i,j,k,1) - (tauxxp+divxp);
                fluxx(i,j,k,2) = fluxx(i,j,k,2) - tauyxp;
                fluxx(i,j,k,3) = fluxx(i,j,k,3) - tauzxp;
                fluxx(i,j,k,4) = fluxx(i,j,k,4) - (half*phiflx + kxp*(prim(i,j,k,4)-prim(i-1,j,k,4))/dx[0]);

                Real meanT = 0.5*(prim(i-1,j,k,4)+prim(i,j,k,4));
                Real meanP = 0.5*(prim(i-1,j,k,5)+prim(i,j,k,5));

                if (algorithm_type_gpu == 2) {

                    // compute dk
                    for (int ns=0; ns<nspecies; ++ns) {
                        Real term1 = (prim(i,j,k,6+nspecies+ns)-prim(i-1,j,k,6+nspecies+ns))/dx[0];
                        meanXk[ns] = 0.5*(prim(i-1,j,k,6+nspecies+ns)+prim(i,j,k,6+nspecies+ns));
                        meanYk[ns] = 0.5*(prim(i-1,j,k,6+ns)+prim(i,j,k,6+ns));
                        Real term2 = (meanXk[ns]-meanYk[ns])*(prim(i,j,k,5)-prim(i-1,j,k,5))/dx[0]/meanP;
                        dk[ns] = term1 + term2;
                        soret[ns] = 0.5*(chi(i-1,j,k,ns)*prim(i-1,j,k,6+nspecies+ns)+chi(i,j,k,ns)*prim(i,j,k,6+nspecies+ns))
                            *(prim(i,j,k,4)-prim(i-1,j,k,4))/dx[0]/meanT;
                    }

                    // compute Fk (based on Eqn. 2.5.24, Giovangigli's book)
                    for (int kk=0; kk<nspecies; ++kk) {
                        Fk[kk] = 0.;
                        for (int ll=0; ll<nspecies; ++ll) {
                            Fk[kk] = Fk[kk] - half*(Dij(i-1,j,k,kk*nspecies_gpu+ll)+Dij(i,j,k,kk*nspecies_gpu+ll))*( dk[ll] +soret[ll]);
                        }
                    }

                    // compute Q (based on Eqn. 2.5.25, Giovangigli's book)
                    GetEnthalpies(meanT,hk,hcp_gpu,nspecies_gpu);

                    Real Q5 = 0.;
                    for (int ns=0; ns<nspecies; ++nspecies) {
                        Q5 = Q5 + (hk[ns] + 0.5 * Runiv*meanT*(chi(i-1,j,k,ns)+chi(i,j,k,ns))/molmass[ns])*Fk[ns];
                    }
                    // heat conduction already included in flux(5)       

                    fluxx(i,j,k,4) = fluxx(i,j,k,4) + Q5;

                    for (int ns=0; ns<nspecies; ++nspecies) {
                        fluxx(i,j,k,5+ns) = fluxx(i,j,k,5+ns) + Fk[ns];
                    }
                }
                 
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

            });
#else
        
        const Box& bx = mfi.tilebox();
        
	diff_flux(ARLIM_3D(bx.loVect()), ARLIM_3D(bx.hiVect()),
		  cons_in[mfi].dataPtr(),  
		  prim_in[mfi].dataPtr(),  
		  eta_in[mfi].dataPtr(),  
		  zeta_in[mfi].dataPtr(),  
		  kappa_in[mfi].dataPtr(),  
		  chi_in[mfi].dataPtr(),  
		  D_in[mfi].dataPtr(),  
		  flux_in[0][mfi].dataPtr(),
		  flux_in[1][mfi].dataPtr(),
#if (AMREX_SPACEDIM == 3)
		  flux_in[2][mfi].dataPtr(),
#endif
		  cornx_in[0][mfi].dataPtr(),
		  cornx_in[1][mfi].dataPtr(),
		  cornx_in[2][mfi].dataPtr(),
		  corny_in[0][mfi].dataPtr(),
		  corny_in[1][mfi].dataPtr(),
		  corny_in[2][mfi].dataPtr(),
		  cornz_in[0][mfi].dataPtr(),
		  cornz_in[1][mfi].dataPtr(),
		  cornz_in[2][mfi].dataPtr(),
		  visccorn_in[mfi].dataPtr(),
		  ZFILL(dx));

#endif
    }

    Real wgt2 = 1./12.;
    Real wgt1 = 0.5 + wgt2;

    // from namelist
    int nspecies_gpu = nspecies;
    int algorithm_type_gpu = algorithm_type;
    int nvars_gpu = nvars;
    int nprimvars_gpu = nprimvars;
    Real Runiv_gpu = Runiv;
    
    // from namelist
    GpuArray<Real,MAX_SPECIES> hcv_gpu;
    for (int n=0; n<nspecies; ++n) {
        hcv_gpu[n] = hcv[n];
    }
    
    GpuArray<Real,MAX_SPECIES> molmass_gpu;
    for (int n=0; n<nspecies; ++n) {
        molmass_gpu[n] = molmass[n];
    }    
    
    // Loop over boxes
    for ( MFIter mfi(cons_in); mfi.isValid(); ++mfi) {

        AMREX_D_TERM(const Array4<Real>& xflux = flux_in[0].array(mfi); ,
                     const Array4<Real>& yflux = flux_in[1].array(mfi); ,
                     const Array4<Real>& zflux = flux_in[2].array(mfi));

        const Array4<const Real> prim = prim_in.array(mfi);
        const Array4<const Real> cons = cons_in.array(mfi);

        const Box& tbx = mfi.nodaltilebox(0);
        const Box& tby = mfi.nodaltilebox(1);
        const Box& tbz = mfi.nodaltilebox(2);

        if (advection_type == 1) { // interpolate primitive quantities
            
            // Loop over the cells and compute fluxes
            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            
                GpuArray<Real,MAX_SPECIES+5> conserved;
                GpuArray<Real,MAX_SPECIES+6> primitive;
                GpuArray<Real,MAX_SPECIES  > Yk;
                    
                for (int l=0; l<nprimvars_gpu; ++l) {
                    primitive[l] = wgt1*(prim(i,j,k,l)+prim(i-1,j,k,l)) - wgt2*(prim(i-2,j,k,l)+prim(i+1,j,k,l));
                }

                Real temp = primitive[4];
                Real rho = primitive[0];
                conserved[0] = rho;

                // want sum of specden == rho
                for (int n=0; n<nspecies_gpu; ++n) {
                    Yk[n] = primitive[6+n];
                }

                Real intenergy;
                GetEnergy(intenergy, Yk, temp, hcv_gpu, nspecies_gpu);

                Real vsqr = primitive[1]*primitive[1] + primitive[2]*primitive[2] + primitive[3]*primitive[3];

                conserved[4] = rho*intenergy + 0.5*rho*vsqr;

                xflux(i,j,k,0) += conserved[0]*primitive[1];
                xflux(i,j,k,1) += conserved[0]*(primitive[1]*primitive[1])+primitive[5];
                xflux(i,j,k,2) += conserved[0]*primitive[1]*primitive[2];
                xflux(i,j,k,3) += conserved[0]*primitive[1]*primitive[3];

                xflux(i,j,k,4) += primitive[1]*conserved[4] + primitive[5]*primitive[1];

                if (algorithm_type_gpu == 2) { // Add advection of concentration
                    for (int n=0; n<nspecies_gpu; ++n) {
                        xflux(i,j,k,5+n) += rho*primitive[6+n]*primitive[1];
                    }
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            
                GpuArray<Real,MAX_SPECIES+5> conserved;
                GpuArray<Real,MAX_SPECIES+6> primitive;
                GpuArray<Real,MAX_SPECIES  > Yk;
                    
                for (int l=0; l<nprimvars_gpu; ++l) {
                    primitive[l] = wgt1*(prim(i,j,k,l)+prim(i,j-1,k,l)) - wgt2*(prim(i,j-2,k,l)+prim(i,j+1,k,l));
                }

                Real temp = primitive[4];
                Real rho = primitive[0];
                conserved[0] = rho;

                // want sum of specden == rho
                for (int n=0; n<nspecies_gpu; ++n) {
                    Yk[n] = primitive[6+n];
                }

                Real intenergy;
                GetEnergy(intenergy, Yk, temp, hcv_gpu, nspecies_gpu);

                Real vsqr = primitive[1]*primitive[1] + primitive[2]*primitive[2] + primitive[3]*primitive[3];

                conserved[4] = rho*intenergy + 0.5*rho*vsqr;

                yflux(i,j,k,0) += conserved[0]*primitive[2];
                yflux(i,j,k,1) += conserved[0]*primitive[1]*primitive[2];
                yflux(i,j,k,2) += conserved[0]*primitive[2]*primitive[2]+primitive[5];
                yflux(i,j,k,3) += conserved[0]*primitive[3]*primitive[2];

                yflux(i,j,k,4) += primitive[2]*conserved[4] + primitive[5]*primitive[2];

                if (algorithm_type_gpu == 2) { // Add advection of concentration
                    for (int n=0; n<nspecies_gpu; ++n) {
                        yflux(i,j,k,5+n) += rho*primitive[6+n]*primitive[2];
                    }
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            
                GpuArray<Real,MAX_SPECIES+5> conserved;
                GpuArray<Real,MAX_SPECIES+6> primitive;
                GpuArray<Real,MAX_SPECIES  > Yk;
                    
                for (int l=0; l<nprimvars_gpu; ++l) {
                    primitive[l] = wgt1*(prim(i,j,k,l)+prim(i,j,k-1,l)) - wgt2*(prim(i,j,k-2,l)+prim(i,j,k+1,l));
                }

                Real temp = primitive[4];
                Real rho = primitive[0];
                conserved[0] = rho;

                // want sum of specden == rho
                for (int n=0; n<nspecies_gpu; ++n) {
                    Yk[n] = primitive[6+n];
                }

                Real intenergy;
                GetEnergy(intenergy, Yk, temp, hcv_gpu, nspecies_gpu);

                Real vsqr = primitive[1]*primitive[1] + primitive[2]*primitive[2] + primitive[3]*primitive[3];

                conserved[4] = rho*intenergy + 0.5*rho*vsqr;

                zflux(i,j,k,0) += conserved[0]*primitive[3];
                zflux(i,j,k,1) += conserved[0]*primitive[1]*primitive[3];
                zflux(i,j,k,2) += conserved[0]*primitive[2]*primitive[3];
                zflux(i,j,k,3) += conserved[0]*primitive[3]*primitive[3]+primitive[5];

                zflux(i,j,k,4) += primitive[3]*conserved[4] + primitive[5]*primitive[3];

                if (algorithm_type_gpu == 2) { // Add advection of concentration
                    for (int n=0; n<nspecies_gpu; ++n) {
                        zflux(i,j,k,5+n) += rho*primitive[6+n]*primitive[3];
                    }
                }

            });
            
        } else if (advection_type == 2) { // interpolate conserved quantitites

            // Loop over the cells and compute fluxes
            amrex::ParallelFor(tbx, tby, tbz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            
                GpuArray<Real,MAX_SPECIES+5> conserved;
                GpuArray<Real,MAX_SPECIES+6> primitive;
                GpuArray<Real,MAX_SPECIES  > Yk;
    
                // interpolate conserved quantities to faces
                for (int l=0; l<nvars_gpu; ++l) {
                    conserved[l] = wgt1*(cons(i,j,k,l)+cons(i-1,j,k,l)) - wgt2*(cons(i-2,j,k,l)+cons(i+1,j,k,l));
                }

                // compute velocities
                for (int l=1; l<4; ++l) {
                    primitive[l] = conserved[l]/conserved[0];
                }

                // want sum of specden == rho
                for (int n=0; n<nspecies_gpu; ++n) {
                    Yk[n] = conserved[5+n]/conserved[0];
                }

                // compute temperature
                Real vsqr = primitive[1]*primitive[1] + primitive[2]*primitive[2] + primitive[3]*primitive[3];
                Real intenergy = conserved[4]/conserved[0] - 0.5*vsqr;
                GetTemperature(intenergy, Yk, primitive[4], nspecies_gpu, hcv_gpu);

                // compute pressure
                GetPressureGas(primitive[5], Yk, conserved[0], primitive[4], nspecies_gpu, Runiv_gpu, molmass_gpu);

                xflux(i,j,k,0) += conserved[0]*primitive[1];
                xflux(i,j,k,1) += conserved[0]*(primitive[1]*primitive[1])+primitive[5];
                xflux(i,j,k,2) += conserved[0]*primitive[1]*primitive[2];
                xflux(i,j,k,3) += conserved[0]*primitive[1]*primitive[3];

                xflux(i,j,k,4) += primitive[1]*conserved[4] + primitive[5]*primitive[1];

                if (algorithm_type_gpu == 2) { // Add advection of concentration
                    for (int n=0; n<nspecies_gpu; ++n) {
                        xflux(i,j,k,5+n) += conserved[5+n]*primitive[1];
                    }
                }
            },

            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            
                GpuArray<Real,MAX_SPECIES+5> conserved;
                GpuArray<Real,MAX_SPECIES+6> primitive;
                GpuArray<Real,MAX_SPECIES  > Yk;
    
                // interpolate conserved quantities to faces
                for (int l=0; l<nvars_gpu; ++l) {
                    conserved[l] = wgt1*(cons(i,j,k,l)+cons(i,j-1,k,l)) - wgt2*(cons(i,j-2,k,l)+cons(i,j+1,k,l));
                }

                // compute velocities
                for (int l=1; l<4; ++l) {
                    primitive[l] = conserved[l]/conserved[0];
                }

                // want sum of specden == rho
                for (int n=0; n<nspecies_gpu; ++n) {
                    Yk[n] = conserved[5+n]/conserved[0];
                }

                // compute temperature
                Real vsqr = primitive[1]*primitive[1] + primitive[2]*primitive[2] + primitive[3]*primitive[3];
                Real intenergy = conserved[4]/conserved[0] - 0.5*vsqr;
                GetTemperature(intenergy, Yk, primitive[4], nspecies_gpu, hcv_gpu);

                // compute pressure
                GetPressureGas(primitive[5], Yk, conserved[0], primitive[4], nspecies_gpu, Runiv_gpu, molmass_gpu);

                yflux(i,j,k,0) += conserved[0]*primitive[2];
                yflux(i,j,k,1) += conserved[0]*primitive[1]*primitive[2];
                yflux(i,j,k,2) += conserved[0]*primitive[2]*primitive[2]+primitive[5];
                yflux(i,j,k,3) += conserved[0]*primitive[3]*primitive[2]  ;
           
                yflux(i,j,k,4) += primitive[2]*conserved[4] + primitive[5]*primitive[2];

                if (algorithm_type_gpu == 2) { // Add advection of concentration
                    for (int n=0; n<nspecies_gpu; ++n) {
                        yflux(i,j,k,5+n) += conserved[5+n]*primitive[2];
                    }
                }
            },
                
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            
                GpuArray<Real,MAX_SPECIES+5> conserved;
                GpuArray<Real,MAX_SPECIES+6> primitive;
                GpuArray<Real,MAX_SPECIES  > Yk;
    
                // interpolate conserved quantities to faces
                for (int l=0; l<nvars_gpu; ++l) {
                    conserved[l] = wgt1*(cons(i,j,k,l)+cons(i,j,k-1,l)) - wgt2*(cons(i,j,k-2,l)+cons(i,j,k+1,l));
                }

                // compute velocities
                for (int l=1; l<4; ++l) {
                    primitive[l] = conserved[l]/conserved[0];
                }

                // want sum of specden == rho
                for (int n=0; n<nspecies_gpu; ++n) {
                    Yk[n] = conserved[5+n]/conserved[0];
                }

                // compute temperature
                Real vsqr = primitive[1]*primitive[1] + primitive[2]*primitive[2] + primitive[3]*primitive[3];
                Real intenergy = conserved[4]/conserved[0] - 0.5*vsqr;
                GetTemperature(intenergy, Yk, primitive[4], nspecies_gpu, hcv_gpu);

                // compute pressure
                GetPressureGas(primitive[5], Yk, conserved[0], primitive[4], nspecies_gpu, Runiv_gpu, molmass_gpu);


                zflux(i,j,k,0) += conserved[0]*primitive[3];
                zflux(i,j,k,1) += conserved[0]*primitive[1]*primitive[3];
                zflux(i,j,k,2) += conserved[0]*primitive[2]*primitive[3];
                zflux(i,j,k,3) += conserved[0]*primitive[3]*primitive[3]+primitive[5];

                zflux(i,j,k,4) += primitive[3]*conserved[4] + primitive[5]*primitive[3];

                if (algorithm_type_gpu == 2) { // Add advection of concentration
                    for (int n=0; n<nspecies_gpu; ++n) {
                        zflux(i,j,k,5+n) += conserved[5+n]*primitive[3];
                    }
                }
            });
            
        }
    }
}
