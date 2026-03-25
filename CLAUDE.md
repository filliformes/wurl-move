# Wurl — Physically-modeled Wurlitzer 200A Electric Piano

## Overview
Port of OpenWurli (hal0zer0/openwurli, GPL-3.0) to single-file C for Schwung on Ableton Move.
Module ID: `wurl`, component_type: `sound_generator`, API: `plugin_api_v2`.

## Signal Chain
Modal Reed Oscillator (7 modes, Euler-Bernoulli beam theory)
→ Electrostatic Pickup (capacitive 1/(1-y) nonlinearity)
→ Attack Noise (bandpassed burst)
→ Preamp (HPF → gain → asymmetric tanh saturation → LPF)
→ Tremolo (5.63 Hz sine LFO → CdS LDR model → preamp gain modulation)
→ Power Amp (Newton-Raphson Class AB with crossover distortion)
→ Speaker Cabinet (Hammerstein polynomial + HPF/LPF + thermal compression)

## Architecture
- 16-voice polyphony, round-robin voice stealing (releasing voices first)
- Per-note: tip mass ratio, reed compliance, eigenvalue interpolation, spatial coupling
- Per-note frequency jitter (Ornstein-Uhlenbeck process)
- All DSP in double precision, output as int16_t stereo

## Parameters (1 page, 8 knobs)
| # | Key         | Label   | Range | Default | Function |
|---|-------------|---------|-------|---------|----------|
| 1 | volume      | Volume  | 0-1   | 0.5     | Output level (squared taper) |
| 2 | tremolo     | Tremolo | 0-1   | 0.5     | LDR modulation depth |
| 3 | speaker     | Speaker | 0-1   | 0.5     | Cabinet character (0=bypass) |
| 4 | attack      | Attack  | 0-1   | 0.5     | Hammer onset time |
| 5 | decay       | Decay   | 0-1   | 0.5     | Note sustain (0=long, 1=percussive) |
| 6 | brightness  | Bright  | 0-1   | 0.5     | Upper partial emphasis |
| 7 | bark        | Bark    | 0-1   | 0.5     | Pickup nonlinearity |
| 8 | tune        | Tune    | 0-1   | 0.5     | ±100 cents detune |

## Build
```bash
./scripts/build.sh     # Docker ARM64 cross-compile
./scripts/install.sh   # SCP to move.local
```

## Files
- `src/dsp/wurl.c` — all DSP (single file)
- `src/module.json` — Schwung module manifest
- `scripts/Dockerfile` — ARM64 cross-compiler
- `scripts/build.sh` / `scripts/install.sh` — build & deploy

## Key Schwung Conventions
- `get_param` returns -1 for unknown keys
- `knob_N_name` / `knob_N_value` / `knob_N_adjust` pattern for knob overlay
- `chain_params` returns JSON array of parameter descriptors
- `state` key for save/restore (space-separated floats)
- Render output: interleaved int16_t stereo, BLOCK_SIZE=128
