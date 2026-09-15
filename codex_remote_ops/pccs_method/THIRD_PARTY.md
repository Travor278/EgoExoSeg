# Third-party provenance

- O-MaMa: https://github.com/Maria-SanVil/O-MaMa at `0f187c65cb9d8f8df1d8b5f445edbb0485956d88`. AGPL-3.0, license copied verbatim to `third_party/O-MaMa/LICENSE`. The `aligned_model.py` adapters reuse and adapt its matching arithmetic and import its original descriptor/model modules; those adapters retain the same AGPL-3.0 terms. No claim is made to relicense the rest of EgoExoSeg.
- DINOv2: https://github.com/facebookresearch/dinov2 at `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Obtain original source and Apache-2.0 notices from that pinned repository; no DINOv2 source/weights are vendored here.
- V2-SAM-O: https://github.com/jaychempan/V2-SAM-O at `0e3bc33dec3e202ffbb86cec01038e60b18c162a`. The supplied patch targets that exact source. Follow its upstream terms and those of the vendored SAM/MMEngine/etc. dependencies; this branch does not replace their licenses.
- scikit-learn/joblib/numpy/scipy: external runtime dependencies, not vendored. The logged training environment and package versions are preserved for reproducibility.

Training feature records include benchmark-derived numeric metrics, not raw imagery or masks. Raw dataset access and redistribution remain subject to the original dataset terms.
