#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>
#include <functional>
#include <string.h>
#include <sstream>
#include <cstddef>
#include <sys/stat.h>
#include <chrono>
#include <bitset>

using namespace std;

// some global variables
int RAND_SAMPLES; // will all be set later (if needed)
int image_m;
int image_n;
int image_o;
int image_d = 1;
float SSD0 = 0.0;
float SSD1 = 0.0;
float beta = 1;
// float SIGMA=8.0;
int qc = 1;

// struct for multi-threading of mind-calculation
struct mind_data
{
    float *im1;
    float *d1;
    uint64_t *mindq;
    int qs;
    int ind_d1;
};

#include "transformations.h"
#include "primsMST.h"
#include "regularisation.h"
#include "MINDSSCbox.h"
#include "dataCostD.h"

void deeds(float *im1, float *im1b, float *warped1, int m, int n, int o, float alpha, int levels, bool verbose)
{
    vector<int> grid_spacing = {8, 7, 6, 5, 4};
    vector<int> search_radius = {8, 7, 6, 5, 4};
    vector<int> quantisation = {5, 4, 3, 2, 1};

    if (!verbose)
        cout.rdbuf(nullptr);

    cout << "Starting registration\n";
    cout << "=============================================================\n";

    RAND_SAMPLES = 1; // fixed/efficient random sampling strategy

    image_m = m;
    image_n = n;
    image_o = o;

    int sz = m * n * o;

    // assume we are working with CT scans (add 1024 HU)
    float thresholdF = -1024;
    float thresholdM = -1024;

    for (int i = 0; i < sz; i++)
    {
        im1b[i] -= thresholdF;
        im1[i] -= thresholdM;
    }

    // READ AFFINE MATRIX from linearBCV if provided (else start from identity)

    float *X = new float[16];

    cout << "Starting with identity transform.\n";
    fill(X, X + 16, 0.0f);
    X[0] = 1.0f;
    X[1 + 4] = 1.0f;
    X[2 + 8] = 1.0f;
    X[3 + 12] = 1.0f;

    // PATCH-RADIUS FOR MIND/SSC DESCRIPTORS

    vector<int> mind_step;
    for (int i = 0; i < quantisation.size(); i++)
    {
        mind_step.push_back(floor(0.5f * (float)quantisation[i] + 1.0f));
    }

    cout << "MIND STEPS: " << mind_step[0] << " " << mind_step[1] << " " << mind_step[2] << " " << mind_step[3] << " " << mind_step[4] << endl;

    int step1 = 0, hw1 = 0;
    float quant1 = 0.0;

    // set initial flow-fields to 0; i indicates backward (inverse) transform
    // u is in x-direction (2nd dimension), v in y-direction (1st dim) and w in z-direction (3rd dim)
    float *ux = new float[sz];
    float *vx = new float[sz];
    float *wx = new float[sz];
    for (int i = 0; i < sz; i++)
    {
        ux[i] = 0.0;
        vx[i] = 0.0;
        wx[i] = 0.0;
    }
    int m2 = 0, n2 = 0, o2 = 0, sz2 = 0;
    int m1 = 0, n1 = 0, o1 = 0, sz1 = 0;

    m2 = m / grid_spacing[0];
    n2 = n / grid_spacing[0];
    o2 = o / grid_spacing[0];
    sz2 = m2 * n2 * o2;
    float *u1 = new float[sz2];
    float *v1 = new float[sz2];
    float *w1 = new float[sz2];
    float *u1i = new float[sz2];
    float *v1i = new float[sz2];
    float *w1i = new float[sz2];
    for (int i = 0; i < sz2; i++)
    {
        u1[i] = 0.0;
        v1[i] = 0.0;
        w1[i] = 0.0;
        u1i[i] = 0.0;
        v1i[i] = 0.0;
        w1i[i] = 0.0;
    }

    float *warped0 = new float[m * n * o];
    warpAffine(warped0, im1, im1b, X, ux, vx, wx);

    uint64_t *im1_mind = new uint64_t[m * n * o];
    uint64_t *im1b_mind = new uint64_t[m * n * o];
    uint64_t *warped_mind = new uint64_t[m * n * o];

    auto time1a = chrono::steady_clock::now();

    //==========================================================================================

    for (int level = 0; level < levels; level++)
    {
        quant1 = quantisation[level];

        float prev = mind_step[max(level - 1, 0)]; // max(min(label_quant[max(level-1,0)],2.0f),1.0f);
        float curr = mind_step[level];             // max(min(label_quant[level],2.0f),1.0f);

        float timeMIND = 0;
        float timeSmooth = 0;
        float timeData = 0;
        float timeTrans = 0;

        if (level == 0 | prev != curr)
        {
            auto time1 = chrono::steady_clock::now();

            descriptor(im1_mind, warped0, m, n, o, mind_step[level]); // im1 affine
            descriptor(im1b_mind, im1b, m, n, o, mind_step[level]);

            auto time2 = chrono::steady_clock::now();

            timeMIND += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
        }

        step1 = grid_spacing[level];
        hw1 = search_radius[level];

        int len3 = pow(hw1 * 2 + 1, 3);
        m1 = m / step1;
        n1 = n / step1;
        o1 = o / step1;
        sz1 = m1 * n1 * o1;

        float *costall = new float[sz1 * len3];
        float *u0 = new float[sz1];
        float *v0 = new float[sz1];
        float *w0 = new float[sz1];
        int *ordered = new int[sz1];
        int *parents = new int[sz1];
        float *edgemst = new float[sz1];

        cout << "==========================================================\n";
        cout << "Level " << level << " grid=" << step1 << " with sizes: " << m1 << "x" << n1 << "x" << o1 << " hw=" << hw1 << " quant=" << quant1 << "\n";
        cout << "==========================================================\n";

        // FULL-REGISTRATION FORWARDS
        auto time1 = chrono::steady_clock::now();

        upsampleDeformationsCL(u0, v0, w0, u1, v1, w1, m1, n1, o1, m2, n2, o2);
        upsampleDeformationsCL(ux, vx, wx, u0, v0, w0, m, n, o, m1, n1, o1);
        // float dist=landmarkDistance(ux,vx,wx,m,n,o,distsmm,casenum);
        warpAffine(warped1, im1, im1b, X, ux, vx, wx);
        u1 = new float[sz1];
        v1 = new float[sz1];
        w1 = new float[sz1];

        auto time2 = chrono::steady_clock::now();

        timeTrans += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
        cout << "T" << flush;

        time1 = chrono::steady_clock::now();

        descriptor(warped_mind, warped1, m, n, o, mind_step[level]);

        time2 = chrono::steady_clock::now();

        timeMIND += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
        cout << "M" << flush;

        time1 = chrono::steady_clock::now();
        dataCostCL(im1b_mind, warped_mind, costall, m, n, o, len3, step1, hw1, quant1, alpha, RAND_SAMPLES);
        time2 = chrono::steady_clock::now();

        timeData += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
        cout << "D" << flush;
        time1 = chrono::steady_clock::now();
        primsGraph(im1b, ordered, parents, edgemst, step1, m, n, o);
        regularisationCL(costall, u0, v0, w0, u1, v1, w1, hw1, step1, quant1, ordered, parents, edgemst);
        time2 = chrono::steady_clock::now();
        timeSmooth += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
        cout << "S" << flush;

        // FULL-REGISTRATION BACKWARDS
        time1 = chrono::steady_clock::now();
        upsampleDeformationsCL(u0, v0, w0, u1i, v1i, w1i, m1, n1, o1, m2, n2, o2);
        upsampleDeformationsCL(ux, vx, wx, u0, v0, w0, m, n, o, m1, n1, o1);
        warpImageCL(warped1, im1b, warped0, ux, vx, wx);
        u1i = new float[sz1];
        v1i = new float[sz1];
        w1i = new float[sz1];
        time2 = chrono::steady_clock::now();
        timeTrans += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
        cout << "T" << flush;
        time1 = chrono::steady_clock::now();
        descriptor(warped_mind, warped1, m, n, o, mind_step[level]);

        time2 = chrono::steady_clock::now();
        timeMIND += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
        cout << "M" << flush;
        time1 = chrono::steady_clock::now();
        dataCostCL(im1_mind, warped_mind, costall, m, n, o, len3, step1, hw1, quant1, alpha, RAND_SAMPLES);
        time2 = chrono::steady_clock::now();
        timeData += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
        cout << "D" << flush;
        time1 = chrono::steady_clock::now();
        primsGraph(warped0, ordered, parents, edgemst, step1, m, n, o);
        regularisationCL(costall, u0, v0, w0, u1i, v1i, w1i, hw1, step1, quant1, ordered, parents, edgemst);
        time2 = chrono::steady_clock::now();
        timeSmooth += chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();
        cout << "S" << flush;

        cout << "\nTime: MIND=" << timeMIND << ", data=" << timeData << ", MST-reg=" << timeSmooth << ", transf.=" << timeTrans << "\n speed=" << 2.0 * (float)sz1 * (float)len3 / (timeData + timeSmooth) << " dof/s\n";

        time1 = chrono::steady_clock::now();
        consistentMappingCL(u1, v1, w1, u1i, v1i, w1i, m1, n1, o1, step1);
        time2 = chrono::steady_clock::now();
        float timeMapping = chrono::duration_cast<chrono::duration<float>>(time2 - time1).count();

        // cout<<"Time consistentMapping: "<<timeMapping<<"  \n";

        // upsample deformations from grid-resolution to high-resolution (trilinear=1st-order spline)
        float jac = jacobian(u1, v1, w1, m1, n1, o1, step1);

        cout << "SSD before registration: " << SSD0 << " and after " << SSD1 << "\n";
        m2 = m1;
        n2 = n1;
        o2 = o1;
        cout << "\n";

        delete u0;
        delete v0;
        delete w0;
        delete costall;

        delete parents;
        delete ordered;
    }
    delete im1_mind;
    delete im1b_mind;
    //==========================================================================================
    //==========================================================================================

    auto time2a = chrono::steady_clock::now();

    float timeALL = chrono::duration_cast<chrono::duration<float>>(time2a - time1a).count();

    upsampleDeformationsCL(ux, vx, wx, u1, v1, w1, m, n, o, m1, n1, o1);

    float *flow = new float[sz1 * 3];
    for (int i = 0; i < sz1; i++)
    {
        flow[i] = u1[i];
        flow[i + sz1] = v1[i];
        flow[i + sz1 * 2] = w1[i];
        // flow[i+sz1*3]=u1i[i]; flow[i+sz1*4]=v1i[i]; flow[i+sz1*5]=w1i[i];
    }

    // WRITE OUTPUT DISPLACEMENT FIELD AND IMAGE
    warpAffine(warped1, im1, im1b, X, ux, vx, wx);

    for (int i = 0; i < sz; i++)
    {
        warped1[i] += thresholdM;
    }

    cout << "SSD before registration: " << SSD0 << " and after " << SSD1 << "\n";

    cout << "Finished. Total time: " << timeALL << " sec." << endl;
}
