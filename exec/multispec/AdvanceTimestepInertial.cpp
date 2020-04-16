
#include "hydro_functions.H"

#include "common_functions.H"

#include "gmres_functions.H"

#include "multispec_functions.H"


#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MultiFabUtil.H>


// argv contains the name of the inputs file entered at the command line
void AdvanceTimestepInertial(std::array< MultiFab, AMREX_SPACEDIM >& umac,
                             MultiFab& rho_old, MultiFab& rho_new,
                             MultiFab& rhotot_old, MultiFab& rhotot_new,
                             MultiFab& pi, MultiFab& eta, 
                             std::array< MultiFab, NUM_EDGE >&  eta_ed,
                             MultiFab& kappa, MultiFab& Temp,
                             std::array< MultiFab, NUM_EDGE >& Temp_ed,
                             MultiFab& diff_mass_fluxdiv,
                             MultiFab& stoch_mass_fluxdiv,
                             std::array< MultiFab, AMREX_SPACEDIM >& stoch_mass_flux,
                             StochMassFlux& sMassFlux,
                             StochMomFlux& sMomFlux,
                             const Real& dt, const Real& time, const int& istep,
                             const Geometry& geom)
{
  
    BL_PROFILE_VAR("AdvanceTimestepInertial()",AdvanceTimestepInertial);

    BoxArray ba = rho_old.boxArray();
    DistributionMapping dmap = rho_old.DistributionMap();

    const Real* dx = geom.CellSize();
    
    Vector<Real> weights;
    weights = {1.0};
    
    Real theta_alpha = 1./dt;

    MultiFab rho_update (ba,dmap,nspecies,0);
    MultiFab gmres_rhs_p(ba,dmap,       1,0);
    MultiFab dpi        (ba,dmap,       1,1);

    std::array< MultiFab, AMREX_SPACEDIM > mold;
    std::array< MultiFab, AMREX_SPACEDIM > mtemp;
    std::array< MultiFab, AMREX_SPACEDIM > adv_mom_fluxdiv;
    std::array< MultiFab, AMREX_SPACEDIM > diff_mom_fluxdiv;
    std::array< MultiFab, AMREX_SPACEDIM > gmres_rhs_v;
    std::array< MultiFab, AMREX_SPACEDIM > dumac;
    std::array< MultiFab, AMREX_SPACEDIM > gradpi;
    std::array< MultiFab, AMREX_SPACEDIM > rhotot_fc_old;
    std::array< MultiFab, AMREX_SPACEDIM > rhotot_fc_new;
    std::array< MultiFab, AMREX_SPACEDIM > rho_fc;
    std::array< MultiFab, AMREX_SPACEDIM > diff_mass_flux;
    std::array< MultiFab, AMREX_SPACEDIM > total_mass_flux;
    std::array< MultiFab, AMREX_SPACEDIM > stoch_mom_fluxdiv;

    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        mold[d]             .define(convert(ba,nodal_flag_dir[d]), dmap,        1, 1);
        mtemp[d]            .define(convert(ba,nodal_flag_dir[d]), dmap,        1, 1);
        adv_mom_fluxdiv[d]  .define(convert(ba,nodal_flag_dir[d]), dmap,        1, 0);
        diff_mom_fluxdiv[d] .define(convert(ba,nodal_flag_dir[d]), dmap,        1, 0);
        gmres_rhs_v[d]      .define(convert(ba,nodal_flag_dir[d]), dmap,        1, 0);
        dumac[d]            .define(convert(ba,nodal_flag_dir[d]), dmap,        1, 1);
        gradpi[d]           .define(convert(ba,nodal_flag_dir[d]), dmap,        1, 0);
        rhotot_fc_old[d]    .define(convert(ba,nodal_flag_dir[d]), dmap,        1, 1);
        rhotot_fc_new[d]    .define(convert(ba,nodal_flag_dir[d]), dmap,        1, 1);
        rho_fc[d]           .define(convert(ba,nodal_flag_dir[d]), dmap, nspecies, 0);
        diff_mass_flux[d]   .define(convert(ba,nodal_flag_dir[d]), dmap, nspecies, 0);
        total_mass_flux[d]  .define(convert(ba,nodal_flag_dir[d]), dmap, nspecies, 0);
        stoch_mom_fluxdiv[d].define(convert(ba,nodal_flag_dir[d]), dmap, nspecies, 0);
    }

    // make copies of old quantities
    // copy umac into umac_tmp if using bds

    //////////////////////////////////////////////
    /// Step 1 - Calculate Predictor Diffusive and Stochastic Fluxes
    //////////////////////////////////////////////

    // diff/stoch_mass_fluxdiv already contain F_i
    // this was already done in Step 0 (initialization) or Step 6 from the previous time step
    
    //////////////////////////////////////////////
    // Step 2 - Predictor Euler Step
    //////////////////////////////////////////////

    /*
    if (use_charged_fluid) {
        // compute old Lorentz force
    }
    */

    // average rho_old and rhotot_old to faces
    AverageCCToFace(rho_old,rho_fc,0,nspecies,1,geom);
    AverageCCToFace(rhotot_old,rhotot_fc_old,0,1,-1,geom);

    // add D^n and St^n to rho_update
    MultiFab::Copy(rho_update,diff_mass_fluxdiv,0,0,nspecies,0);
    if (variance_coef_mass != 0.) {
        MultiFab::Add(rho_update,stoch_mass_fluxdiv,0,0,nspecies,0);
    }

    // add A^n to rho_update
    if (advection_type >= 1) {
        Abort("AdvanceTimestepInterial: bds not supported");
    }
    else {
        MkAdvSFluxdiv(umac,rho_fc,rho_update,geom,0,nspecies,true);
    }
   
    // set rho_new = rho_old + dt * (A^n + D^n + St^n)
    MultiFab::LinComb(rho_new,1.,rho_old,0,dt,rho_update,0,0,nspecies,0);

    // compute rhotot from rho in VALID REGION
    ComputeRhotot(rho_new,rhotot_new);

    // fill rho and rhotot ghost cells
    FillRhoRhototGhost(rho_new,rhotot_new,geom);

    // average rho_new and rhotot_new to faces
    AverageCCToFace(rho_new,rho_fc,0,nspecies,1,geom);
    AverageCCToFace(rhotot_new,rhotot_fc_new,0,1,-1,geom);
    
    /*
    if (use_charged_fluid) {
        // compute total charge
        // compute permittivity
    }
    */

    //////////////////////////////////////////////
    // Step 3 - Calculate Corrector Diffusive and Stochastic Fluxes
    // Step 4 - Predictor Crank-Nicolson Step
    //////////////////////////////////////////////

    // compute mold
    ConvertMToUmac(rhotot_fc_old,umac,mold,0);

    // build up rhs_v for gmres solve: first set gmres_rhs_v to mold/dt
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        MultiFab::Copy(gmres_rhs_v[d],mold[d],0,0,1,0);
        gmres_rhs_v[d].mult(1/dt,0);
    }
        
    // compute grad pi^n
    ComputeGrad(pi,gradpi,0,0,1,geom);

    /*
    if (barodiffusion_type == 2) {
       // barodiffusion uses lagged grad(pi)
    }
    else if (barodiffusion_type == 3) {
       // compute p0 from rho0*g
    }
    */

    // subtract grad pi^n from gmres_rhs_v
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        MultiFab::Subtract(gmres_rhs_v[d],gradpi[d],0,0,1,0);
    }

    // compute adv_mom_fluxdiv = A^n for momentum
    MkAdvMFluxdiv(umac,mold,adv_mom_fluxdiv,dx,0);

    // add A^n for momentum into gmres_rhs_v
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        MultiFab::Add(gmres_rhs_v[d],adv_mom_fluxdiv[d],0,0,1,0);
    }

    // compute diff_mom_fluxdiv = A_0^n v^n
    MkDiffusiveMFluxdiv(diff_mom_fluxdiv,umac,eta,eta_ed,kappa,geom,dx,0);
    
    // add (1/2) A_0^n v^n to gmres_rhs_v
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        MultiFab::Saxpy(gmres_rhs_v[d],0.5,diff_mom_fluxdiv[d],0,0,1,0);
    }

    if (variance_coef_mom != 0.) {

        // fill the stochastic multifabs with a new set of random numbers
        sMomFlux.fillMomStochastic();

       // compute and save stoch_mom_fluxdiv = div(Sigma^n) (save for later)
//       call stochastic_m_fluxdiv(mla,the_bc_tower%bc_tower_array,stoch_mom_fluxdiv,.false., &
//                                 eta,eta_ed,Temp,Temp_ed,dx,dt,weights)

           // add div(Sigma^n) to gmres_rhs_v
//          do i=1,dm
//                    call multifab_plus_plus_c(gmres_rhs_v(n,i),1,stoch_mom_fluxdiv(n,i),1,1,0)
//          end do

           }
    
    
    
}