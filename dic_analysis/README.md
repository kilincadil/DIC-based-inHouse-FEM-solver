# DIC-based pixel-wise yield stress and hardening identification

Two-stage pipeline that turns a tensile-test image sequence into pixel-wise maps
of yield onset, yield stress, and local hardening coefficient.

## Pipeline

| Order | Script | Input | Output |
|---|---|---|---|
| 1 | `dic_displacement_fields.py` | `ref.tif`, `mask.png`, `traction/*` | `U_a100/U_<i>.npy`, `V_a100/V_<i>.npy` |
| 2 | `yield_stress_hardening.py` | `U_a100/`, `V_a100/` | `S_p_yield_onset_frame.npy`, `yield_stress_vm.npy`, `yield_stress_dev.npy`, `K_hardening_coef_filled.npy` |

Run them in order — step 2 reads what step 1 writes.

### 1. `dic_displacement_fields.py`

Digital Image Correlation via OpenCV's DIS optical flow with variational
refinement (Brox et al., ECCV 2004). For each image of the traction sequence it
computes the dense displacement field relative to the reference image, applies
the binary mask, and saves the X/Y displacements cropped to the region of
interest.

### 2. `yield_stress_hardening.py`

For every pixel:

1. **Equivalent strain history** — strain from the displacement gradients,
   deviatoric decomposition under plane stress, von Mises equivalent strain
   `Evm`; corrupted load steps are interpolated and the stack is smoothed along
   the load-step axis.
2. **Yield onset** (`S_p`) — the elastic regime is fitted over a growing window;
   the onset frame is the first local maximum of R2, i.e. where the linear fit
   stops improving.
3. **Yield stress** — von Mises stress evaluated at each pixel's own onset
   frame, both from plane-stress Hooke's law (`yield_stress_vm.npy`) and from
   the deviatoric equivalent strain, `sigma = 3 G Evm` (`yield_stress_dev.npy`).
4. **Hardening coefficient** (`K`) — post-yield slope of the `Evm` curve after
   removing the extrapolated elastic trend, normalised by the macroscopic slope
   `SLOPE_REF`. Implausible values are masked and filled from their 8-neighbours.

## Configuration

Both scripts have a `CONFIG` block at the top — data paths, material properties
(`E`, `nu`), optical flow parameters, ROI crop, and analysis thresholds. Nothing
below that block needs editing for a new dataset.

Key values currently set:

- `E = 205000 MPa`, `nu = 0.3`
- Optical flow: `alpha = 100`, `delta = 1`, `gamma = 0`, `epsilon = 0.002`,
  finest scale 0, patch size 4, stride 1
- ROI crop: rows 400–4000, cols 1211–4311
- Valid yield onset frames: 2–42
- `SLOPE_REF = 4.46e-05` (macroscopic tensile curve)
- Hardening coefficient clipped to 0–3

## OpenCV requirement

The variational-refinement setters used in step 1
(`setVariationalRefinementAlpha` etc.) are not exposed by the standard
`opencv-python` wheel — a local OpenCV build is required. Point
`OPENCV_LIB_DIR` / `OPENCV_BIN_DIR` at that build, or set
`USE_CUSTOM_OPENCV = False` to use the pip-installed `cv2` if your version
supports them.

## Dependencies

```bash
pip install -r requirements.txt
```

`numba` (JIT + parallel loops) and `joblib` (threaded stress computation) do the
heavy lifting; the first run of the numba-compiled functions includes a
compilation delay.

## Credits

- Step 1 optical flow: original by Eddidoune (2021)
- Step 2 identification: original by Qi Hu
- Restructured by kilincadil
