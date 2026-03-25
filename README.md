# Wurl

Physically-modeled Wurlitzer 200A electric piano for [Ableton Move](https://www.ableton.com/move/),
built for the [Schwung](https://github.com/charlesvestal/schwung) framework.

Port of [OpenWurli](https://github.com/hal0zer0/openwurli) by hal0zer0 to single-file C.

## Features

- Complete 200A signal chain modeled from first principles (no samples)
- 7-mode modal reed oscillator with per-note Euler-Bernoulli beam physics
- Electrostatic pickup with 1/(1-y) capacitive nonlinearity ("bark")
- Tremolo that modulates timbre via emitter feedback (not just volume)
- Class AB power amp with crossover distortion
- Speaker cabinet with polynomial nonlinearity and thermal compression
- 16-voice polyphony with voice stealing
- 8 expressive parameters mapped to Move's knobs

## Controls

|Knob      |Function                                             |
|----------|-----------------------------------------------------|
|Volume    |Output level (audio taper, like the real 200A pot)   |
|Tremolo   |Depth of 5.63 Hz tremolo (timbral modulation)        |
|Speaker   |Cabinet character: 0% = flat bypass, 100% = authentic|
|Attack    |Reed onset time (instant → slow ring-up)             |
|Decay     |Sustain length (long → very percussive)              |
|Brightness|Upper harmonic content (dark → bright/metallic)      |
|Bark      |Pickup nonlinearity (clean bell → aggressive growl)  |
|Tune      |Detune ±100 cents                                    |

## Building

```
./scripts/build.sh
```

Requires Docker or an `aarch64-linux-gnu-gcc` cross-compiler.

## Installation

```
./scripts/install.sh
```

Or install via the Module Store in Schwung.

## Credits

- **hal0zer0** — [OpenWurli](https://github.com/hal0zer0/openwurli) (original Rust implementation)
- **OldBassMan** (Freesound) — Wurlitzer 200A recordings used for OpenWurli calibration
- **Robbert van der Helm** — nih-plug framework (original plugin)

## License

GPL-3.0 — see [LICENSE](LICENSE)
