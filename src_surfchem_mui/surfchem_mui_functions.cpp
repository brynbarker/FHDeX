#include "surfchem_mui_functions.H"
#include "AMReX_ParmParse.H"

AMREX_GPU_MANAGED int surfchem_mui::nspec_mui;

AMREX_GPU_MANAGED GpuArray<amrex::Real, MAX_SPECIES> surfchem_mui::mom_inertia;

void InitializeSurfChemMUINamespace()
{
    // extract inputs parameters
    ParmParse pp;

    // number of species involved in mui (via adsorption/desorption)
    pp.get("nspec_mui",nspec_mui);

    // get moment of inertia (used only for diatomic molecules with dof = 5 and e0 = 0)
    std::vector<amrex::Real> mi_tmp(MAX_SPECIES);
    pp.getarr("mom_inertia",mi_tmp,0,nspec_mui);
    for (int n=0; n<nspec_mui; n++) mom_inertia[n] = mi_tmp[n];

    return;
}

void mui_push(MultiFab& cu, MultiFab& prim, const amrex::Real* dx, mui::uniface2d &uniface, const int step)
// this routine pushes the following information to MUI
// - species number densities and temperature of FHD cells contacting the interface
// it assumes that the interface is perpendicular to the z-axis
// and includes cells with the smallest value of z (i.e. k=0)
{
    for (MFIter mfi(cu,false); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Dim3 lo = lbound(bx);
        Dim3 hi = ubound(bx);
        const Array4<Real> & cu_arr = cu.array(mfi);
        const Array4<Real> & prim_arr = prim.array(mfi);

        // unless bx contains cells at the interface, skip 
        int k = 0;
        if (k<lo.z || k>hi.z) continue;

        for (int i = lo.x; i<=hi.x; ++i)
        {
            for (int j = lo.y; j<= hi.y; ++j)
            {
                double x = prob_lo[0]+(i+0.5)*dx[0];
                double y = prob_lo[1]+(j+0.5)*dx[1];

                std::string channel;

                for (int n = 0; n < nspec_mui; ++n)
                {
                    channel = "CH_density";
                    channel += '0'+(n+1);   // assuming nspec_mui<10

                    double dens = cu_arr(i,j,k,5+n);    // mass density
                    dens *= AVONUM/molmass[n];          // number density

                    uniface.push(channel,{x,y},dens);
                }

                //channel = "CH_temp";

                //uniface.push(channel,{x,y},prim_arr(i,j,k,4));
            }
        }
    }

    uniface.commit(step);

    return;
}

void mui_fetch(MultiFab& cu, MultiFab& prim, const amrex::Real* dx, mui::uniface2d &uniface, const int step)
// this routine fetches the following information from MUI:
// - adsoprtion and desoprtion counts of each species between time points
// it assumes that the interface is perpendicular to the z-axis
// and includes cells with the smallest value of z (i.e. k=0)
{
    mui::sampler_kmc_fhd2d<int> s({dx[0],dx[1]});
    mui::chrono_sampler_exact2d t;

    for (MFIter mfi(cu,false); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Dim3 lo = lbound(bx);
        Dim3 hi = ubound(bx);
        const Array4<Real> & cu_arr = cu.array(mfi);
        const Array4<Real> & prim_arr = prim.array(mfi);

        // unless bx contains cells at the interface, skip 
        // ad-hoc fix to avoid memory leakage
        int k = 0;
        if (k<lo.z || k>hi.z)
        {
            double x = prob_lo[0]+(lo.x+0.5)*dx[0];
            double y = prob_lo[1]+(lo.y+0.5)*dx[1];

            uniface.fetch("CH_ac1",{x,y},step,s,t);

            continue;
        }

        for (int i = lo.x; i<=hi.x; ++i)
        {
            for (int j = lo.y; j<= hi.y; ++j)
            {
                double x = prob_lo[0]+(i+0.5)*dx[0];
                double y = prob_lo[1]+(j+0.5)*dx[1];
                double dV = dx[0]*dx[1]*dx[2];
                //double temp = prim_arr(i,j,k,4);
                double temp = t_lo[2];

                for (int n = 0; n < nspec_mui; ++n)
                {
                    std::string channel;
                    int ac,dc;

                    channel = "CH_ac";
                    channel += '0'+(n+1);   // assuming nspec_mui<10
                    ac = uniface.fetch(channel,{x,y},step,s,t);

                    channel = "CH_dc";
                    channel += '0'+(n+1);   // assuming nspec_mui<10
                    dc = uniface.fetch(channel,{x,y},step,s,t);

                    double mass = molmass[n]/AVONUM;
                    double kBTm = k_B*temp/mass;
                    double sqrtkBTm = sqrt(kBTm);
                    double vx,vy,vz;
                    double dmomx,dmomy,dmomz,derg;

                    double kBTI = k_B*temp/mom_inertia[n];
                    double sqrtkBTI = sqrt(kBTI);
                    double omegax,omegay;

                    dmomx = dmomy = dmomz = derg = 0.;

                    // sample incoming velocities and compute translational energy change
                    for (int l=0;l<ac;l++)
                    {
                        vx = RandomNormal(0.,sqrtkBTm);
                        vy = RandomNormal(0.,sqrtkBTm);
                        vz = -sqrt(-2.*kBTm*log(1.-Random()));

                        dmomx -= mass*vx;
                        dmomy -= mass*vy;
                        dmomz += mass*vz;   // note the sign
                        derg  -= 0.5*mass*(vx*vx+vy*vy+vz*vz);
                    }

                    // sample outgoing velocities and compute translational energy change
                    for (int l=0;l<dc;l++)
                    {
                        vx = RandomNormal(0.,sqrtkBTm);
                        vy = RandomNormal(0.,sqrtkBTm);
                        vz = sqrt(-2.*kBTm*log(1.-Random()));

                        dmomx += mass*vx;
                        dmomy += mass*vy;
                        dmomz += mass*vz;
                        derg  += 0.5*mass*(vx*vx+vy*vy+vz*vz);
                    }

                    // sample non-translational energy change
                    if (dof[n]==5 && e0[n]==0.)
                    {
                        // in this case (i.e. diatomic molecules), non-translational energy = rotational energy
                        for (int l=0;l<ac;l++)
                        {
                            // angular velocity (diatomic)
                            omegax = RandomNormal(0.,sqrtkBTI);
                            omegay = RandomNormal(0.,sqrtkBTI);
                            derg -= 0.5*mom_inertia[n]*(omegax*omegax+omegay*omegay);
                        }

                        for (int l=0;l<dc;l++)
                        {
                            // angular velocity (diatomic)
                            omegax = RandomNormal(0.,sqrtkBTI);
                            omegay = RandomNormal(0.,sqrtkBTI);
                            derg += 0.5*mom_inertia[n]*(omegax*omegax+omegay*omegay);
                        }
                    }
                    else
                    {
                        amrex::Abort("Sampling of non-translational energy is implemented only for dof=5 and e0 = 0");
                    }

                    cu_arr(i,j,k,0) += (dc-ac)*mass/dV;
                    cu_arr(i,j,k,5+n) += (dc-ac)*mass/dV;

                    cu_arr(i,j,k,1) += dmomx/dV;
                    cu_arr(i,j,k,2) += dmomy/dV;
                    cu_arr(i,j,k,3) += dmomz/dV;
                    cu_arr(i,j,k,4) += derg/dV;
                }
            }
        }
    }

    uniface.forget(step);

    return;
}

void mui_announce_send_recv_span(mui::uniface2d &uniface,MultiFab& mf,const Real* dx)
{
    // find the lo and hi points of a square that covers all boxes assigned to the MPI process

    int lox,loy,loz,hix,hiy,hiz;

    bool isfirst = true;

    for (MFIter mfi(mf,false); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        Dim3 lo = lbound(bx);
        Dim3 hi = ubound(bx);

        if (isfirst)
        {
            lox = lo.x;
            loy = lo.y;
            loz = lo.z;
            hix = hi.x;
            hiy = hi.y;
            hiz = hi.z;

            isfirst = false;
        }
        else
        {
            lox = (lox<lo.x) ? lox : lo.x;
            loy = (loy<lo.y) ? loy : lo.y;
            loz = (loz<lo.z) ? loz : lo.z;
            hix = (hix>hi.x) ? hix : hi.x;
            hiy = (hiy>hi.y) ? hiy : hi.y;
            hiz = (hiz>hi.z) ? hiz : hi.z;
        }
    }

    // announce span to MUI
    // we assume that the FHD layer contacting the KMC is k = 0
    int k = 0;

    if (k>=loz && k<=hiz)
    {
        double tmp[2];

        tmp[0] = prob_lo[0] + lox*dx[0];
        tmp[1] = prob_lo[1] + loy*dx[1];
        point<double,2> span_lo(tmp);

        tmp[0] = prob_lo[0] + (hix+1)*dx[0];
        tmp[1] = prob_lo[1] + (hiy+1)*dx[1];
        point<double,2> span_hi(tmp);

        mui::geometry::box<config_2d> span(span_lo,span_hi);

        uniface.announce_send_span(0.,(double)max_step,span);
        uniface.announce_recv_span(0.,(double)max_step,span);
    }
    else
    {
        double tmp[2];

        tmp[0] = -1.;
        tmp[1] = -1.;
        point<double,2> span_lo(tmp);

        tmp[0] = -0.9;
        tmp[1] = -0.9;
        point<double,2> span_hi(tmp);

        mui::geometry::box<config_2d> span(span_lo,span_hi);

        uniface.announce_send_span(0.,(double)max_step,span);
        uniface.announce_recv_span(0.,(double)max_step,span);
    }

    return;
}
