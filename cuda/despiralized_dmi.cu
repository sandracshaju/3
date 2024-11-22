#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

//Despiralized Exchange +  Despiralzied Dzyaloshinskii-Moriya interaction for bulk material.

// Energy:
// E = (A nabla m)² + D sin(qy) (MydzMz - MzdzMy +MydxMx -MxdxMz) + D cos(qy) (MzdxMy - MydxMz + MydxMx -Mxdzmy) + D²/4A My²
// q=D/2A
// Effective field:
//
// 	Hx = 2A/Bs nabla²Mx + 2D/Bs sin(qy) dyMy + 2D/Bs cos(qy) dzMy
// 	Hy = 2A/Bs nabla²My - 2D/Bs sin(qy) (dzMz + dxMx) - 2D/Bs cos(qy) (dzMx - dxMz) - D²/2ABs My
// 	Hz = 2A/Bs nabla²Mz + 2D/Bs sin(qy) dzMy - 2D/Bs cos(qy) dxMy
//
// Boundary conditions:
//
// 	        2A dxMx = 0
// 	 D Mz + 2A dxMy = 0
// 	-D My + 2A dxMz = 0
//
// 	-D Mz + 2A dyMx = 0
// 	        2A dyMy = 0
// 	 D Mx + 2A dyMz = 0
//
// 	 D My + 2A dzMx = 0
// 	-D Mx + 2A dzMy = 0
// 	        2A dzMz = 0
//
extern "C" __global__ void
adddmibulk(float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,
           float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
           float* __restrict__ Ms_, float Ms_mul,
           float* __restrict__ aLUT2d, float* __restrict__ DLUT2d,
           uint8_t* __restrict__ regions,
           float cx, float cy, float cz, int Nx, int Ny, int Nz, uint8_t PBC, uint8_t OpenBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I = idx(ix, iy, iz);                      // central cell index
    float3 h = make_float3(0.0,0.0,0.0);          // add to H
    float3 m0 = make_float3(mx[I], my[I], mz[I]); // central m
    uint8_t r0 = regions[I];
    int i_;                                       // neighbor index

    if(is0(m0)) {
        return;
    }

    // x derivatives (along length)
    {
        float3 m1 = make_float3(0.0f, 0.0f, 0.0f);     // left neighbor
        i_ = idx(lclampx(ix-1), iy, iz);               // load neighbor m if inside grid, keep 0 otherwise
        if (ix-1 >= 0 || PBCx) {
            m1 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m1)? r0 : regions[i_];
        float A = aLUT2d[symidx(r0, r1)];
        float D = DLUT2d[symidx(r0, r1)];
        float D_2A = D/(2.0f*A);
        if (!is0(m1) || !OpenBC){                      // do nothing at an open boundary
            if (is0(m1)) {                             // neighbor missing
                m1.x = m0.x;
                m1.y = m0.y - (-cx * D_2A * m0.z);
                m1.z = m0.z + (-cx * D_2A * m0.y);
            }
            h   += (2.0f*A/(cx*cx)) * (m1 - m0);       // exchange
            h.x += (D/cx)*sin(D_2A*cy*iy)*(-m1.y);
            h.y -= (D/cx)*(sin(D_2A*cy*iy)*(-m1.x) - cos(D_2A*cy*iy)*(-m1.z));
            h.z -= (D/cx)*cos(D_2A*cy*iy)*(-m1.y);
        }
    }


    {
        float3 m2 = make_float3(0.0f, 0.0f, 0.0f);     // right neighbor
        i_ = idx(hclampx(ix+1), iy, iz);
        if (ix+1 < Nx || PBCx) {
            m2 = make_float3(mx[i_], my[i_], mz[i_]);
        }
        int r1 = is0(m2)? r0 : regions[i_];
        float A = aLUT2d[symidx(r0, r1)];
        float D = DLUT2d[symidx(r0, r1)];
        float D_2A = D/(2.0f*A);
        if (!is0(m2) || !OpenBC){
            if (is0(m2)) {
                m2.x = m0.x;
                m2.y = m0.y - (+cx * D_2A * m0.z);
                m2.z = m0.z + (+cx * D_2A * m0.y);
            }
            h   += (2.0f*A/(cx*cx)) * (m2 - m0);
            h.x += (D/cx)*sin(D_2A*cy*iy)*(m2.y);
            h.y -= (D/cx)*(sin(D_2A*cy*iy)*(m2.x) - cos(D_2A*cy*iy)*(m2.z));
            h.z -= (D/cx)*cos(D_2A*cy*iy)*(m2.y);
        }
    }

    
    // z derivatives
    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        {
            float3 m1 = make_float3(0.0f, 0.0f, 0.0f);
            i_ = idx(ix, iy, lclampz(iz-1));
            if (iz-1 >= 0 || PBCz) {
                m1 = make_float3(mx[i_], my[i_], mz[i_]);
            }
            int r1 = is0(m1)? r0 : regions[i_];
            float A = aLUT2d[symidx(r0, r1)];
            float D = DLUT2d[symidx(r0, r1)];
            float D_2A = D/(2.0f*A);
            if (!is0(m1) || !OpenBC){
                if (is0(m1)) {
                    m1.x = m0.x - (-cz * D_2A * m0.y);
                    m1.y = m0.y + (-cz * D_2A * m0.x);
                    m1.z = m0.z;
                }
                h   += (2.0f*A/(cz*cz)) * (m1 - m0);
                h.x += (D/cz)*cos(D_2A*cy*iy)*(-m1.y);
                h.y -= (D/cz)*(sin(D_2A*cy*iy)*(-m1.z) + cos(D_2A*cy*iy)*(-m1.z));
                h.z += (D/cz)*sin(D_2A*cy*iy)*(-m1.y);
            }
        }

        // top neighbor
        {
            float3 m2 = make_float3(0.0f, 0.0f, 0.0f);
            i_ = idx(ix, iy, hclampz(iz+1));
            if (iz+1 < Nz || PBCz) {
                m2 = make_float3(mx[i_], my[i_], mz[i_]);
            }
            int r1 = is0(m2)? r0 : regions[i_];
            float A = aLUT2d[symidx(r0, r1)];
            float D = DLUT2d[symidx(r0, r1)];
            float D_2A = D/(2.0f*A);
            if (!is0(m2) || !OpenBC){
                if (is0(m2)) {
                    m2.x = m0.x - (+cz * D_2A * m0.y);
                    m2.y = m0.y + (+cz * D_2A * m0.x);
                    m2.z = m0.z;
                }
                h   += (2.0f*A/(cz*cz)) * (m2 - m0);
                h.x += (D/cz)*cos(D_2A*cy*iy)*(m2.y);
                h.y -= (D/cz)*(sin(D_2A*cy*iy)*(m2.z) + cos(D_2A*cy*iy)*(m2.z));
                h.z += (D/cz)*sin(D_2A*cy*iy)*(m2.y);
            }
        }
    }

    // write back, result is H + Hdmi + Hex
    float invMs = inv_Msat(Ms_, Ms_mul, I);
    Hx[I] += h.x*invMs;
    Hy[I] += h.y*invMs;
    Hz[I] += h.z*invMs;
}

// Note on boundary conditions.
//
// We need the derivative and laplacian of m in point A, but e.g. C lies out of the boundaries.
// We use the boundary condition in B (derivative of the magnetization) to extrapolate m to point C:
// 	m_C = m_A + (dm/dx)|_B * cellsize
//
// When point C is inside the boundary, we just use its actual value.
//
// Then we can take the central derivative in A:
// 	(dm/dx)|_A = (m_C - m_D) / (2*cellsize)
// And the laplacian:
// 	lapl(m)|_A = (m_C + m_D - 2*m_A) / (cellsize^2)
//
// All these operations should be second order as they involve only central derivatives.
//
//    ------------------------------------------------------------------ *
//   |                                                   |             C |
//   |                                                   |          **   |
//   |                                                   |        ***    |
//   |                                                   |     ***       |
//   |                                                   |   ***         |
//   |                                                   | ***           |
//   |                                                   B               |
//   |                                               *** |               |
//   |                                            ***    |               |
//   |                                         ****      |               |
//   |                                     ****          |               |
//   |                                  ****             |               |
//   |                              ** A                 |               |
//   |                         *****                     |               |
//   |                   ******                          |               |
//   |          *********                                |               |
//   |D ********                                         |               |
//   |                                                   |               |
//   +----------------+----------------+-----------------+---------------+
//  -1              -0.5               0               0.5               1
//                                 x
