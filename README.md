# Wurl

Physically-modeled Wurlitzer 200A electric piano for [Ableton Move](https://www.ableton.com/move/),
built for the [Schwung](https://github.com/charlesvestal/schwung) open-source plugin framework.

Port of [OpenWurli](https://github.com/hal0zer0/openwurli) by **hal0zer0** (GPL-3.0) to single-file C,
with additional features for the Move platform.

## Signal Chain

Every stage of the Wurlitzer 200A is modeled from first principles — no samples, no impulse responses.

```
Reed Oscillator (7 modal modes, Euler-Bernoulli beam theory)
  → Electrostatic Pickup (capacitive 1/(1-y) nonlinearity — the source of "bark")
  → Attack Noise (bandpassed burst on hammer strike)
  ──── voice sum ────
  → [2x Oversampling] ─────────────────────────────────────────────────
  │  → Preamp (HPF → gain × LDR tremolo modulation → asymmetric tanh)  │
  │  → Volume (squared audio taper)                                     │
  │  → Power Amp (Newton-Raphson Class AB with crossover distortion)    │
  │  → Speaker Cabinet (polynomial + HPF/LPF + thermal compression)     │
  │  → Soft limiter                                                     │
  → [Downsample] ──────────────────────────────────────────────────────
  → Darken (smooth one-pole LPF)
  → Spring Reverb (dispersive allpass chain)
  → Output (int16 stereo)
```

## Controls

### Knobs (8)

| Knob       | Function |
|------------|----------|
| Volume     | Output level (squared audio taper) |
| Tremolo    | Depth of 5.63 Hz tremolo (timbral, not just volume) |
| Attack     | Reed onset speed: 0% = slow ring-up, 100% = fast/punchy |
| Decay      | Note sustain: 0% = short/percussive, 100% = long sustain |
| Brightness | Upper harmonic content: 0% = very dark, 50% = neutral, 100% = bright |
| Darken     | Smooth low-pass filter: 0% = bypass, 100% = very dark (800 Hz) |
| Bark       | Pickup nonlinearity: clean bell → aggressive growl |
| Reverb     | Spring reverb macro (controls mix + decay + damping together) |

### Menu Parameters (jog wheel)

| Parameter | Function |
|-----------|----------|
| Speaker   | Cabinet character: 0% = flat bypass, 100% = authentic 200A |
| Tune      | Detune ±100 cents |
| Preset    | Cycle through 10 built-in presets (applied in real time) |

### Presets

| # | Name | Character |
|---|------|-----------|
| 0 | Classic 200A | Neutral starting point, no reverb |
| 1 | Dreamy Keys | Slow trem, warm, darkened, lush reverb |
| 2 | Barky Soul | High bark, punchy, bright, authentic speaker |
| 3 | Surf Spring | Deep trem + heavy spring reverb |
| 4 | Dark Ballad | Dark, mellow, long sustain, gentle reverb |
| 5 | Percussive Clav | Instant attack, very short, bright, dry |
| 6 | Warm Pad | Slow attack, very long sustain, big reverb |
| 7 | Lo-Fi Tape | Very dark, full speaker, slight detune |
| 8 | Bright Bell | Clean (no speaker), bright harmonics, reverb |
| 9 | Gospel Growl | Maximum bark + speaker, aggressive |

## What's Ported from OpenWurli

The core physical model is faithfully ported from [OpenWurli](https://github.com/hal0zer0/openwurli):

- **Modal reed oscillator** — 7-mode quadrature oscillator with Ornstein-Uhlenbeck frequency jitter, cosine onset ramp, and progressive damper
- **Per-note physics** — tip mass ratio, eigenvalue interpolation, reed compliance, spatial coupling (Simpson integration of mode shapes), velocity S-curve
- **Electrostatic pickup** — time-varying RC circuit with bilinear discretization, C(y) = C0/(1-y) nonlinearity
- **Attack noise** — bandpassed burst on hammer strike
- **MLP v2 corrections** — 2→8→8→11 neural network (195 weights) runs once at note-on, correcting H2-H6 frequencies, decay rates, and displacement scale. Trained on real Wurlitzer recordings.
- **Power amplifier** — behavioral Newton-Raphson Class AB model with crossover distortion
- **Speaker cabinet** — Hammerstein polynomial waveshaper (BL asymmetry + Kms hardening) with tanh excursion limiting, thermal voice coil compression, HPF/LPF
- **CdS LDR tremolo model** — asymmetric attack/release envelope, power-law resistance curve
- **Per-note variation** — deterministic FNV hash detuning, mode amplitude offsets, register trim
- **Voice stealing** — 5ms linear crossfade on steal (releasing voices first, then oldest)
- **NaN guards** — on voice sum and final output, with filter state reset

### What's Different from OpenWurli

| Feature | OpenWurli (Rust) | Wurl (C port) |
|---------|-----------------|---------------|
| Preamp | DK method 12-node MNA circuit solver (2750 lines generated) | Simplified 2-stage CE: HPF → gain → asymmetric tanh → LPF |
| Tremolo oscillator | Melange-generated Twin-T circuit (2100 lines) | Behavioral sine LFO (same CdS LDR model) |
| Oversampling | 2x on preamp only | 2x on entire output chain (preamp + power amp + speaker) |
| Polyphony | 64 voices | 16 voices (Move CPU budget) |
| Parameters | 4 (Volume, Tremolo, Speaker, MLP on/off) | 11 (8 knobs + 3 menu), plus 10 presets |
| MIDI timing | Sample-accurate event splitting | Block-rate (128 samples) |
| Spring reverb | None | Dispersive allpass model (Parker 2011) |
| Darken filter | None | One-pole LPF (800 Hz → 20 kHz) |
| Preset system | None | 10 built-in presets with real-time switching |

### What's Not Ported

- **DK method preamp** — the melange-generated 12-node Modified Nodal Analysis circuit solver is 2750 lines of Rust with matrix constants. Too complex to port; replaced with a behavioral approximation.
- **Twin-T tremolo oscillator** — the melange-generated 7-node circuit model is 2100 lines. Replaced with sine LFO (the original's `legacy-tremolo` feature flag).
- **Melange power amp circuit** — the 20-node 7-BJT generated circuit option. The behavioral NR model IS ported.

## Building

```bash
./scripts/build.sh      # Docker ARM64 cross-compile
./scripts/install.sh    # SCP to move.local
```

Requires Docker Desktop or an `aarch64-linux-gnu-gcc` cross-compiler.

## Credits

- **hal0zer0** — [OpenWurli](https://github.com/hal0zer0/openwurli): original Rust implementation of the Wurlitzer 200A physical model (GPL-3.0)
- **OldBassMan** (Freesound) — Wurlitzer 200A recordings used for OpenWurli's MLP training data
- **Robbert van der Helm** — [nih-plug](https://github.com/robbert-vdh/nih-plug) framework (original plugin host)
- **Julian Parker** — "[Efficient Dispersion Generation Structures for Spring Reverb Emulation](https://asp-eurasipjournals.springeropen.com/articles/10.1155/2011/646134)" (2011) — spring reverb algorithm reference
- **Charles Vestal** — [Schwung](https://github.com/charlesvestal/schwung) plugin framework for Ableton Move

## License

GPL-3.0 — inherits from [OpenWurli](https://github.com/hal0zer0/openwurli).

See [LICENSE](LICENSE) for full text.
