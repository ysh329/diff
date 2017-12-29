    for (int p = 0; p < aI_width; p += 8) {
        CL_ELEM_TYPE aa = vload4(0, aI + row * aI_width + p),
                     bb = vload4(0, bT + row * aI_width + p);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb;
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;

        aa = vload4(0, aI + row * aI_width + p+4);
        bb = vload4(0, bT + row * aI_width + p+4);

        c00 += (CL_ELEM_TYPE)aa.s0 * bb;
        c10 += (CL_ELEM_TYPE)aa.s1 * bb;
        c20 += (CL_ELEM_TYPE)aa.s2 * bb;
        c30 += (CL_ELEM_TYPE)aa.s3 * bb;

    }
