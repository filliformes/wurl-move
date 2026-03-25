/**
 * Wurl — Physically-modeled Wurlitzer 200A Electric Piano
 * Schwung sound generator for Ableton Move
 *
 * Port of OpenWurli (hal0zer0, GPL-3.0) to single-file C.
 * Signal chain: Modal reed -> Pickup -> Preamp -> Tremolo -> PowerAmp -> Speaker
 *
 * 16-voice polyphony with round-robin voice stealing.
 *
 * Author: fillioning (port), Original: hal0zer0/openwurli
 * License: GPL-3.0
 * API: plugin_api_v2_t
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define SAMPLE_RATE     44100.0f
#define SAMPLE_RATE_D   44100.0
#define NUM_MODES       7
#define NUM_VOICES      16
#define BLOCK_SIZE      128
#define MIDI_LO         33
#define MIDI_HI         96

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define TWO_PI (2.0 * M_PI)

/* Pickup */
#define PICKUP_TAU          (287.0e3 * 240.0e-12)
#define PICKUP_SENSITIVITY  1.8375
#define PICKUP_MAX_Y        0.98

/* Tremolo */
#define TREM_RATE_HZ        5.63
#define LDR_ATTACK_TAU      0.003
#define LDR_RELEASE_TAU     0.050
#define LDR_R_MIN           50.0
#define LDR_R_MAX           1000000.0
#define LDR_GAMMA           1.1

/* Power amp */
#define PA_OPEN_LOOP_GAIN   19000.0
#define PA_FEEDBACK_BETA    (220.0 / (220.0 + 15000.0))
#define PA_HEADROOM         22.0
#define PA_CROSSOVER_VT     0.013
#define PA_QUIESCENT_GAIN   0.1
#define PA_NR_MAX_ITER      8
#define PA_NR_TOL           1e-6

/* Speaker */
#define SPK_HPF_AUTH_HZ     95.0
#define SPK_HPF_Q           0.75
#define SPK_LPF_AUTH_HZ     5500.0
#define SPK_LPF_Q           0.707
#define SPK_HPF_BYPASS_HZ   20.0
#define SPK_LPF_BYPASS_HZ   20000.0
#define SPK_THERMAL_TAU     5.0

/* Reed jitter */
#define JITTER_SIGMA        0.0004
#define JITTER_TAU_S        0.020
#define JITTER_SUBSAMPLE    16
#define RENORM_INTERVAL     1024
#define SQRT_3              1.7320508080

/* Preamp simplified */
#define PREAMP_GAIN         66.0
#define PREAMP_CIN_FC       329.0
#define PREAMP_BW_FC        15500.0
#define PREAMP_SAT_POS      5.3
#define PREAMP_SAT_NEG      1.0

/* Output */
#define POST_SPEAKER_GAIN   3.3497

static const double BASE_MODE_AMPS[NUM_MODES] =
    {1.0, 0.005, 0.0035, 0.0018, 0.0011, 0.0007, 0.0005};

/* ── Eigenvalue table ─────────────────────────────────────────────── */
#define N_EIG_ROWS 8
static const double EIG_MU[N_EIG_ROWS] = {0.00,0.01,0.05,0.10,0.15,0.20,0.30,0.50};
static const double EIG_BETAS[N_EIG_ROWS][NUM_MODES] = {
    {1.8751,4.6941,7.8548,10.9955,14.1372,17.2788,20.4204},
    {1.8584,4.6849,7.8504,10.9930,14.1356,17.2776,20.4195},
    {1.7920,4.6477,7.8316,10.9830,14.1288,17.2726,20.4158},
    {1.7227,4.6024,7.8077,10.9700,14.1198,17.2660,20.4110},
    {1.6625,4.5618,7.7859,10.9580,14.1114,17.2598,20.4065},
    {1.6097,4.5254,7.7659,10.9470,14.1036,17.2540,20.4023},
    {1.5201,4.4620,7.7310,10.9280,14.0894,17.2434,20.3946},
    {1.3853,4.3601,7.6745,10.8970,14.0650,17.2252,20.3814},
};

/* ── Helpers ──────────────────────────────────────────────────────── */
static double clamp_d(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static double midi_to_freq(int midi) {
    return 440.0 * pow(2.0, (midi - 69) / 12.0);
}

/* ── Per-note tables (ported from tables.rs) ─────────────────────── */
static double tip_mass_ratio(int midi) {
    static const double a[][2] = {{33,0.10},{52,0.00},{62,0.00},{74,0.02},{96,0.01}};
    double m = (double)midi;
    if (m <= a[0][0]) return a[0][1];
    if (m >= a[4][0]) return a[4][1];
    for (int i = 0; i < 4; i++) {
        if (m <= a[i+1][0]) {
            double t = (m - a[i][0]) / (a[i+1][0] - a[i][0]);
            return a[i][1] + t * (a[i+1][1] - a[i][1]);
        }
    }
    return 0.0;
}

static void eigenvalues(double mu, double out[NUM_MODES]) {
    double mu_c = clamp_d(mu, 0.0, 0.50);
    int lo = 0;
    for (int i = N_EIG_ROWS - 1; i >= 0; i--) {
        if (EIG_MU[i] <= mu_c) { lo = i; break; }
    }
    int hi = (lo + 1 < N_EIG_ROWS) ? lo + 1 : N_EIG_ROWS - 1;
    double t = (EIG_MU[hi] > EIG_MU[lo]) ?
        (mu_c - EIG_MU[lo]) / (EIG_MU[hi] - EIG_MU[lo]) : 0.0;
    for (int i = 0; i < NUM_MODES; i++)
        out[i] = EIG_BETAS[lo][i] + t * (EIG_BETAS[hi][i] - EIG_BETAS[lo][i]);
}

static void mode_ratios(double mu, double out[NUM_MODES]) {
    double betas[NUM_MODES];
    eigenvalues(mu, betas);
    double b1_sq = betas[0] * betas[0];
    for (int i = 0; i < NUM_MODES; i++)
        out[i] = (betas[i] * betas[i]) / b1_sq;
}

static double reed_length_mm(int midi) {
    double n = clamp_d((double)(midi - 32), 1.0, 64.0);
    double inches = (n <= 20.0) ? (3.0 - n / 20.0) : (2.0 - (n - 20.0) / 44.0);
    return inches * 25.4;
}

static void reed_blank_dims(int midi, double *w_mm, double *t_mm) {
    int reed = midi - 32;
    if (reed < 1) reed = 1; if (reed > 64) reed = 64;
    double w_in = (reed<=14)?0.151:(reed<=20)?0.127:(reed<=42)?0.121:(reed<=50)?0.111:0.098;
    double t_in;
    if (reed <= 16) t_in = 0.026;
    else if (reed <= 26) { t_in = 0.026 + ((reed-16.0)/10.0)*(0.034-0.026); }
    else t_in = 0.034;
    *w_mm = w_in * 25.4; *t_mm = t_in * 25.4;
}

static double reed_compliance(int midi) {
    double l = reed_length_mm(midi);
    double w, t; reed_blank_dims(midi, &w, &t);
    return (l*l*l) / (w*t*t*t);
}

static double pickup_displacement_scale(int midi) {
    double c = reed_compliance(midi), c_ref = reed_compliance(60);
    return clamp_d(0.75 * pow(c / c_ref, 0.75), 0.02, 0.82);
}

static double fundamental_decay_rate(int midi) {
    double d = 0.005 * pow(midi_to_freq(midi), 1.22);
    return (d > 3.0) ? d : 3.0;
}

static void mode_decay_rates(int midi, const double ratios[NUM_MODES], double out[NUM_MODES]) {
    double base = fundamental_decay_rate(midi);
    for (int i = 0; i < NUM_MODES; i++) out[i] = base * ratios[i] * ratios[i];
}

/* Spatial coupling (Simpson integration of mode shapes) */
static void spatial_coupling(double mu, double reed_len, double out[NUM_MODES]) {
    double betas[NUM_MODES]; eigenvalues(mu, betas);
    double ell_l = clamp_d(6.0 / reed_len, 0.0, 1.0);
    double k_raw[NUM_MODES];
    for (int m = 0; m < NUM_MODES; m++) {
        double beta = betas[m];
        double tip = cosh(beta)-cos(beta)-((cosh(beta)+cos(beta))/(sinh(beta)+sin(beta)))*(sinh(beta)-sin(beta));
        if (fabs(tip) < 1e-30 || ell_l < 1e-12) { k_raw[m]=1.0; continue; }
        double xi0 = 1.0 - ell_l, h = ell_l / 32.0;
        double sig = (cosh(beta)+cos(beta))/(sinh(beta)+sin(beta));
        double sum = 0.0;
        for (int j = 0; j <= 32; j++) {
            double bx = beta*(xi0+j*h);
            double phi = cosh(bx)-cos(bx)-sig*(sinh(bx)-sin(bx));
            double w = (j==0||j==32)?1.0:((j%2==1)?4.0:2.0);
            sum += w * phi;
        }
        k_raw[m] = clamp_d(fabs((sum*h/3.0)/(ell_l*tip)), 0.0, 1.0);
    }
    double k1 = k_raw[0];
    for (int i = 0; i < NUM_MODES; i++)
        out[i] = (k1>1e-30) ? clamp_d(k_raw[i]/k1, 0.0, 1.0) : 1.0;
}

static double velocity_scurve(double vel) {
    double k=1.5, s=1.0/(1.0+exp(-k*(vel-0.5)));
    double s0=1.0/(1.0+exp(k*0.5)), s1=1.0/(1.0+exp(-k*0.5));
    return (s-s0)/(s1-s0);
}

static double velocity_exponent(int midi) {
    double m=(double)midi, t=exp(-0.5*pow((m-62.0)/15.0,2));
    return 1.3+t*0.4;
}

static double pickup_rms_proxy(double ds, double f0, double fc) {
    if (ds<1e-10) return 0.0;
    double r=(1.0-sqrt(1.0-ds*ds))/ds, inv=1.0/sqrt(1.0-ds*ds), sum=0.0, rn=r;
    for (int n=1;n<=8;n++) {
        double cn=2.0*rn*inv, nf=n*f0, h=nf/sqrt(nf*nf+fc*fc);
        sum+=(cn*h)*(cn*h); rn*=r;
    }
    return sqrt(sum);
}

static double register_trim_db(int midi) {
    static const double a[][2]={{36,-1.3},{40,0},{44,-1.3},{48,0.7},{52,0.2},
        {56,-1.0},{60,0},{64,0.9},{68,1.2},{72,0},{76,1.8},{80,2.4},{84,3.6}};
    double m=(double)midi;
    if (m<=a[0][0]) return a[0][1]; if (m>=a[12][0]) return a[12][1];
    for (int i=0;i<12;i++) {
        if (m<=a[i+1][0]) { double t=(m-a[i][0])/(a[i+1][0]-a[i][0]); return a[i][1]+t*(a[i+1][1]-a[i][1]); }
    }
    return 0.0;
}

static double output_scale(int midi, double vel) {
    double ds=pickup_displacement_scale(midi), f0=midi_to_freq(midi);
    double sv=velocity_scurve(vel), vs=pow(sv,velocity_exponent(midi)), vc4=pow(sv,velocity_exponent(60));
    double ed=fmax(ds*vs,1e-6), edr=fmax(0.75*vc4,1e-6);
    double rms=pickup_rms_proxy(ed,f0,2312.0), rr=pickup_rms_proxy(edr,midi_to_freq(60),2312.0);
    double fdb=-20.0*log10(rms/rr), vdb=-0.04*fmax((double)midi-60.0,0.0);
    return pow(10.0,(-35.0+fdb+vdb+register_trim_db(midi)*pow(vel,1.3))/20.0);
}

static void dwell_attenuation(double vel, double f0, const double r[NUM_MODES], double out[NUM_MODES]) {
    double td=clamp_d((0.75+0.25*(1.0-vel))/f0,0.0003,0.020), ss=128.0;
    for (int i=0;i<NUM_MODES;i++) { double ft=f0*r[i]*td; out[i]=exp(-ft*ft/(2.0*ss)); }
    double a0=out[0]; if (a0>1e-30) for (int i=0;i<NUM_MODES;i++) out[i]/=a0;
}

static double onset_ramp_time(double vel, double f0) {
    return clamp_d((2.5+2.5*(1.0-vel))/f0, 0.002, 0.030);
}

static double hash_f64(int midi, uint32_t seed) {
    uint32_t h=2166136261u; h^=(uint32_t)midi; h*=16777619u;
    h^=seed; h*=16777619u; h^=h>>16; h*=2654435769u;
    return (double)(h&0x00FFFFFF)/16777216.0;
}

static double freq_detune(int midi) {
    return 1.0 + (hash_f64(midi,0xDEAD)*2.0-1.0)*0.00173;
}

/* ── Biquad (Audio EQ Cookbook, DF-II Transposed) ─────────────────── */
typedef struct { double b0,b1,b2,a1,a2,z1,z2; } biquad_t;

static void biquad_reset(biquad_t *f) { f->z1=f->z2=0.0; }

static void biquad_set_hpf(biquad_t *f, double fc, double q, double sr) {
    double w0=TWO_PI*fc/sr, alpha=sin(w0)/(2.0*q), cs=cos(w0), a0=1.0+alpha;
    f->b0=((1.0+cs)/2.0)/a0; f->b1=-(1.0+cs)/a0; f->b2=f->b0;
    f->a1=(-2.0*cs)/a0; f->a2=(1.0-alpha)/a0;
}

static void biquad_set_lpf(biquad_t *f, double fc, double q, double sr) {
    double w0=TWO_PI*fc/sr, alpha=sin(w0)/(2.0*q), cs=cos(w0), a0=1.0+alpha;
    f->b0=((1.0-cs)/2.0)/a0; f->b1=(1.0-cs)/a0; f->b2=f->b0;
    f->a1=(-2.0*cs)/a0; f->a2=(1.0-alpha)/a0;
}

static void biquad_set_bpf(biquad_t *f, double fc, double q, double sr) {
    double w0=TWO_PI*fc/sr, alpha=sin(w0)/(2.0*q), a0=1.0+alpha;
    f->b0=alpha/a0; f->b1=0.0; f->b2=-alpha/a0;
    f->a1=(-2.0*cos(w0))/a0; f->a2=(1.0-alpha)/a0;
}

static inline double biquad_process(biquad_t *f, double x) {
    double y=f->b0*x+f->z1;
    f->z1=f->b1*x-f->a1*y+f->z2;
    f->z2=f->b2*x-f->a2*y;
    return y;
}

/* ── Modal Reed Oscillator ───────────────────────────────────────── */
typedef struct {
    double s,c,cos_inc,sin_inc,phase_inc,amplitude,decay_mult,envelope;
    double jitter_drift,damper_rate,damper_mult;
} reed_mode_t;

typedef struct {
    reed_mode_t modes[NUM_MODES];
    uint64_t sample, onset_ramp_samples;
    double onset_ramp_inc, onset_shape_exp;
    int damper_active, damper_ramp_done;
    double damper_ramp_samples, damper_release_count;
    uint32_t jitter_state;
    double jitter_revert, jitter_diffusion;
} reed_t;

static inline double lcg_uniform_scaled(uint32_t *s) {
    *s = (*s)*1664525u+1013904223u;
    return ((double)(*s>>1)/(double)0x7FFFFFFFu*2.0-1.0)*SQRT_3;
}

static void reed_init(reed_t *r, double f0, const double ratios[NUM_MODES],
    const double amps[NUM_MODES], const double decays_db[NUM_MODES],
    double onset_time, double velocity, double sr, uint32_t seed) {
    double dt=1.0/sr;
    r->jitter_revert=exp(-dt/JITTER_TAU_S);
    r->jitter_diffusion=JITTER_SIGMA*sqrt(1.0-r->jitter_revert*r->jitter_revert);
    r->jitter_state=(seed>0)?seed:1;
    r->sample=0; r->damper_active=0; r->damper_ramp_done=0;
    r->damper_ramp_samples=0; r->damper_release_count=0;
    uint32_t js=r->jitter_state;
    for (int i=0;i<NUM_MODES;i++) {
        js=js*1664525u+1013904223u; double u1=(double)(js>>1)/(double)0x7FFFFFFFu;
        js=js*1664525u+1013904223u; double u2=(double)(js>>1)/(double)0x7FFFFFFFu;
        double freq=f0*ratios[i], pi=TWO_PI*freq/sr;
        double an=decays_db[i]/8.686;
        reed_mode_t *m=&r->modes[i];
        m->s=0; m->c=1; m->cos_inc=cos(pi); m->sin_inc=sin(pi);
        m->phase_inc=pi; m->amplitude=amps[i]; m->decay_mult=exp(-an/sr);
        m->envelope=1.0; m->damper_rate=0; m->damper_mult=1.0;
        m->jitter_drift=JITTER_SIGMA*sqrt(-2.0*log(fmax(u1,1e-30)))*cos(TWO_PI*u2);
    }
    r->jitter_state=js;
    uint64_t rs=(uint64_t)(onset_time*sr+0.5);
    r->onset_ramp_samples=rs;
    r->onset_ramp_inc=(rs>0)?(M_PI/(double)rs):0.0;
    r->onset_shape_exp=1.0+(1.0-velocity);
}

static void reed_start_damper(reed_t *r, int midi, double sr) {
    if (midi>=92) return;
    double base=fmax(55.0*pow(2.0,(midi-60.0)/24.0),0.5);
    for (int m=0;m<NUM_MODES;m++) {
        double f=fmin(base*pow(3.0,m),2000.0);
        r->modes[m].damper_rate=f/sr;
        r->modes[m].damper_mult=exp(-r->modes[m].damper_rate);
    }
    r->damper_ramp_samples=((midi<48)?0.050:(midi<72)?0.025:0.008)*sr;
    r->damper_active=1; r->damper_release_count=0; r->damper_ramp_done=0;
}

static void reed_render(reed_t *r, double *out, int frames) {
    for (int i=0;i<frames;i++) {
        double ramp=1.0;
        if (r->sample < r->onset_ramp_samples) {
            double tn=(double)r->sample*r->onset_ramp_inc;
            ramp=pow(0.5*(1.0-cos(tn)), r->onset_shape_exp);
        }
        double df=1.0;
        if (r->damper_active && !r->damper_ramp_done) {
            r->damper_release_count+=1.0;
            if (r->damper_release_count>=r->damper_ramp_samples) r->damper_ramp_done=1;
            else df=r->damper_release_count/r->damper_ramp_samples;
        }
        if ((r->sample%JITTER_SUBSAMPLE)==0)
            for (int m=0;m<NUM_MODES;m++)
                r->modes[m].jitter_drift=r->jitter_revert*r->modes[m].jitter_drift
                    +r->jitter_diffusion*lcg_uniform_scaled(&r->jitter_state);
        double sum=0.0;
        for (int m=0;m<NUM_MODES;m++) {
            reed_mode_t *md=&r->modes[m];
            double di=md->phase_inc*md->jitter_drift;
            double ci=md->cos_inc+(-md->sin_inc*di), si=md->sin_inc+(md->cos_inc*di);
            double sn=md->s*ci+md->c*si, cn=md->c*ci-md->s*si;
            md->s=sn; md->c=cn;
            md->envelope*=md->decay_mult;
            if (r->damper_active) {
                if (r->damper_ramp_done)
                    md->envelope*=md->damper_mult;
                else
                    md->envelope*=exp(-md->damper_rate*df);
            }
            /* Denormal flush */
            if (md->envelope < 1e-15) md->envelope = 0.0;
            sum+=md->s*md->amplitude*md->envelope;
        }
        if ((r->sample%RENORM_INTERVAL)==0 && r->sample>0)
            for (int m=0;m<NUM_MODES;m++) {
                reed_mode_t *md=&r->modes[m];
                double mag=sqrt(md->s*md->s+md->c*md->c);
                if (mag>1e-30) { md->s/=mag; md->c/=mag; }
            }
        out[i]+=sum*ramp;
        r->sample++;
    }
}

static int reed_is_silent(const reed_t *r) {
    double total=0.0;
    for (int m=0;m<NUM_MODES;m++) {
        double lv=r->modes[m].amplitude*r->modes[m].envelope;
        total+=lv*lv;
    }
    return (10.0*log10(total+1e-30) < -80.0);
}

/* ── Pickup (time-varying RC) ────────────────────────────────────── */
typedef struct { double q,beta,ds; } pickup_t;

static void pickup_init(pickup_t *p, double sr, double ds) {
    p->q=1.0; p->beta=(1.0/sr)/(2.0*PICKUP_TAU); p->ds=ds;
}

static void pickup_process(pickup_t *p, double *buf, int frames) {
    for (int i=0;i<frames;i++) {
        double y=clamp_d(buf[i]*p->ds,-PICKUP_MAX_Y,PICKUP_MAX_Y);
        double omy=1.0-y, alpha=p->beta*omy;
        double qn=(p->q*(1.0-alpha)+2.0*p->beta)/(1.0+alpha);
        p->q=qn; buf[i]=(qn*omy-1.0)*PICKUP_SENSITIVITY;
    }
}

/* ── Attack Noise ────────────────────────────────────────────────── */
typedef struct { double amp,decay; uint32_t rem; biquad_t bpf; uint32_t rng; } noise_t;

static void noise_init(noise_t *n, double vel, double f0, double sr, uint32_t seed) {
    n->amp=0.025*vel*vel; n->decay=exp(-1.0/(0.003*sr));
    n->rem=(uint32_t)(0.015*sr); memset(&n->bpf,0,sizeof(biquad_t));
    biquad_set_bpf(&n->bpf,clamp_d(f0*5.0,200.0,2000.0),0.7,sr); n->rng=seed;
}

static void noise_render(noise_t *n, double *buf, int frames) {
    int c=(n->rem<(uint32_t)frames)?(int)n->rem:frames;
    for (int i=0;i<c;i++) {
        n->rng=n->rng*1664525u+1013904223u;
        buf[i]+=n->amp*biquad_process(&n->bpf,(double)(int32_t)(n->rng)/(double)0x7FFFFFFF);
        n->amp*=n->decay;
    }
    n->rem-=(uint32_t)c;
}

/* ── Voice ───────────────────────────────────────────────────────── */
typedef struct {
    reed_t reed; pickup_t pickup; noise_t noise;
    double post_gain; int midi_note, active, releasing; uint64_t age;
} voice_t;

static void voice_note_on(voice_t *v, int note, double vel, double sr, uint32_t seed,
    double brightness, double bark_mod, double attack_mod, double decay_mod, double tune_cents) {
    double mu=tip_mass_ratio(note), ratios[NUM_MODES];
    mode_ratios(mu, ratios);
    double f0=midi_to_freq(note)*freq_detune(note)*pow(2.0,tune_cents/1200.0);
    double dwell[NUM_MODES]; dwell_attenuation(vel,f0,ratios,dwell);
    double amps[NUM_MODES]; double coupling[NUM_MODES];
    spatial_coupling(mu, reed_length_mm(note), coupling);
    for (int i=0;i<NUM_MODES;i++) {
        double var=1.0+(hash_f64(note,0xBEEF+(uint32_t)i)*2.0-1.0)*0.08;
        amps[i]=BASE_MODE_AMPS[i]*dwell[i]*var*coupling[i];
    }
    double vs=pow(velocity_scurve(vel),velocity_exponent(note));
    for (int i=0;i<NUM_MODES;i++) amps[i]*=vs;
    /* Brightness: scale upper modes */
    double bs=0.2+brightness*1.8;
    for (int i=1;i<NUM_MODES;i++) amps[i]*=bs;
    /* Decay rates with decay_mod control */
    double decays[NUM_MODES]; mode_decay_rates(note,ratios,decays);
    /* decay_mod: 0=long sustain (0.3x), 0.5=normal, 1=short/percussive (3x) */
    double dm = 0.3 + decay_mod * 5.4;
    if (dm < 0.3) dm = 0.3; if (dm > 5.7) dm = 5.7;
    for (int i=0;i<NUM_MODES;i++) decays[i]*=dm;
    /* Onset */
    double base_onset=onset_ramp_time(vel,f0);
    double onset=clamp_d(base_onset*(0.2+attack_mod*3.8),0.001,0.100);
    reed_init(&v->reed,f0,ratios,amps,decays,onset,vel,sr,seed);
    /* Pickup with bark control */
    double ds=pickup_displacement_scale(note)*(0.3+bark_mod*1.4);
    pickup_init(&v->pickup,sr,clamp_d(ds,0.02,0.95));
    noise_init(&v->noise,vel,f0,sr,seed);
    v->post_gain=output_scale(note,vel);
    v->midi_note=note; v->active=1; v->releasing=0;
}

static void voice_note_off(voice_t *v) {
    reed_start_damper(&v->reed,v->midi_note,SAMPLE_RATE_D);
    v->releasing=1;
}

static void voice_render(voice_t *v, double *buf, int frames) {
    memset(buf,0,frames*sizeof(double));
    reed_render(&v->reed,buf,frames);
    if (v->noise.rem>0) noise_render(&v->noise,buf,frames);
    pickup_process(&v->pickup,buf,frames);
    for (int i=0;i<frames;i++) buf[i]*=v->post_gain;
}

/* ── Preamp (simplified two-stage CE) ────────────────────────────── */
typedef struct { biquad_t hpf,lpf; double gain,base_gain; } preamp_t;

static void preamp_init(preamp_t *p, double sr) {
    memset(p,0,sizeof(preamp_t));
    biquad_set_hpf(&p->hpf,PREAMP_CIN_FC,0.707,sr);
    biquad_set_lpf(&p->lpf,PREAMP_BW_FC,0.707,sr);
    p->base_gain=PREAMP_GAIN; p->gain=PREAMP_GAIN;
}

static void preamp_set_ldr(preamp_t *p, double r_path) {
    double re1=33000.0, re_eff=(re1*r_path)/(re1+r_path);
    double re_nom=(re1*1000000.0)/(re1+1000000.0);
    p->gain=p->base_gain*clamp_d(re_eff/re_nom,0.1,2.0);
}

static inline double preamp_process(preamp_t *p, double in) {
    double x=biquad_process(&p->hpf,in)*p->gain;
    x=(x>0)?tanh(x/PREAMP_SAT_POS)*PREAMP_SAT_POS:tanh(x/PREAMP_SAT_NEG)*PREAMP_SAT_NEG;
    return biquad_process(&p->lpf,x);
}

/* ── Tremolo (sine LFO + CdS LDR) ───────────────────────────────── */
typedef struct {
    double phase,phase_inc,depth,ldr_env,ldr_atk,ldr_rel,r_series;
} tremolo_t;

static void tremolo_init(tremolo_t *t, double depth, double sr) {
    t->phase=0; t->phase_inc=TWO_PI*TREM_RATE_HZ/sr; t->depth=depth;
    t->ldr_env=0; t->ldr_atk=exp(-1.0/(LDR_ATTACK_TAU*sr));
    t->ldr_rel=exp(-1.0/(LDR_RELEASE_TAU*sr));
    t->r_series=18000.0+50000.0*(1.0-depth);
}

static void tremolo_set_depth(tremolo_t *t, double d) {
    t->depth=clamp_d(d,0.0,1.0);
    t->r_series=18000.0+50000.0*(1.0-t->depth);
}

static double tremolo_process(tremolo_t *t) {
    double lfo=sin(t->phase); t->phase+=t->phase_inc;
    if (t->phase>=TWO_PI) t->phase-=TWO_PI;
    double led=(lfo>0?lfo:0)*t->depth;
    double coeff=(led>t->ldr_env)?t->ldr_atk:t->ldr_rel;
    t->ldr_env=led+coeff*(t->ldr_env-led);
    double drv=clamp_d(t->ldr_env,0.0,1.0), r_ldr;
    if (drv<1e-6) r_ldr=LDR_R_MAX;
    else r_ldr=exp(log(LDR_R_MAX)+(log(LDR_R_MIN)-log(LDR_R_MAX))*pow(drv,LDR_GAMMA));
    return t->r_series+r_ldr;
}

/* ── Power Amp (behavioral NR, Class AB) ─────────────────────────── */
typedef struct { double cl_gain; } power_amp_t;

static void pa_init(power_amp_t *pa) {
    pa->cl_gain=PA_OPEN_LOOP_GAIN/(1.0+PA_OPEN_LOOP_GAIN*PA_FEEDBACK_BETA);
}

static inline double pa_process(power_amp_t *pa, double in) {
    double y=clamp_d(in*pa->cl_gain,-PA_HEADROOM+PA_NR_TOL,PA_HEADROOM-PA_NR_TOL);
    for (int it=0;it<PA_NR_MAX_ITER;it++) {
        double err=in-PA_FEEDBACK_BETA*y, v=PA_OPEN_LOOP_GAIN*err;
        double vs=v*v, vts=PA_CROSSOVER_VT*PA_CROSSOVER_VT, et=exp(-vs/vts);
        double cg=PA_QUIESCENT_GAIN+(1.0-PA_QUIESCENT_GAIN)*(1.0-et);
        double vc=v*cg, dcg=cg+v*(1.0-PA_QUIESCENT_GAIN)*(2.0*v/vts)*et;
        double ta=tanh(vc/PA_HEADROOM), fv=PA_HEADROOM*ta, fd=(1.0-ta*ta)*dcg;
        double res=y-fv, jac=1.0+PA_OPEN_LOOP_GAIN*PA_FEEDBACK_BETA*fd;
        double delta=res/jac; y-=delta;
        if (fabs(delta)<PA_NR_TOL) break;
    }
    return clamp_d(y/PA_HEADROOM, -1.0, 1.0);
}

/* ── Speaker (polynomial + HPF/LPF + thermal) ───────────────────── */
typedef struct {
    biquad_t hpf,lpf; double character,a2,a3,th_coeff,th_alpha,th_state;
} speaker_t;

static void spk_init(speaker_t *s, double sr) {
    memset(s,0,sizeof(speaker_t)); s->character=1.0;
    biquad_set_hpf(&s->hpf,SPK_HPF_AUTH_HZ,SPK_HPF_Q,sr);
    biquad_set_lpf(&s->lpf,SPK_LPF_AUTH_HZ,SPK_LPF_Q,sr);
    s->a2=0.2; s->a3=0.6; s->th_coeff=2.0;
    s->th_alpha=1.0/(SPK_THERMAL_TAU*sr); s->th_state=0;
}

static void spk_set_char(speaker_t *s, double ch, double sr) {
    double c=clamp_d(ch,0,1); if (fabs(c-s->character)<0.002) return;
    s->character=c;
    biquad_set_hpf(&s->hpf,SPK_HPF_BYPASS_HZ*pow(SPK_HPF_AUTH_HZ/SPK_HPF_BYPASS_HZ,c),SPK_HPF_Q,sr);
    biquad_set_lpf(&s->lpf,SPK_LPF_BYPASS_HZ*pow(SPK_LPF_AUTH_HZ/SPK_LPF_BYPASS_HZ,c),SPK_LPF_Q,sr);
    s->a2=0.2*c; s->a3=0.6*c; s->th_coeff=2.0*c;
}

static inline double spk_process(speaker_t *s, double in) {
    double x2=in*in, x3=x2*in;
    double sh=(in+s->a2*x2+s->a3*x3)/(1.0+s->a2+s->a3);
    double lim=(s->character<0.001)?sh:tanh(sh);
    s->th_state+=(x2-s->th_state)*s->th_alpha;
    double tg=1.0/(1.0+s->th_coeff*sqrt(s->th_state));
    return biquad_process(&s->lpf,biquad_process(&s->hpf,lim*tg));
}

/* ── Instance ────────────────────────────────────────────────────── */
typedef struct {
    voice_t voices[NUM_VOICES]; uint64_t age_counter;
    preamp_t preamp; tremolo_t tremolo; power_amp_t pa; speaker_t speaker;
    double vbuf[BLOCK_SIZE], sbuf[BLOCK_SIZE];
    float p_vol,p_trem,p_spk,p_atk,p_dcy,p_brt,p_bark,p_tune;
    float s_vol,s_trem,s_spk; float pitch_bend;
} wurl_t;

typedef struct { const char *key; const char *label; float min,max,step; } knob_def_t;
static const knob_def_t KNOBS[8]={
    {"volume","Volume",0,1,0.01f},{"tremolo","Tremolo",0,1,0.01f},
    {"speaker","Speaker",0,1,0.01f},{"attack","Attack",0,1,0.01f},
    {"decay","Decay",0,1,0.01f},{"brightness","Bright",0,1,0.01f},
    {"bark","Bark",0,1,0.01f},{"tune","Tune",0,1,0.01f}
};

/* ── Lifecycle ───────────────────────────────────────────────────── */
static void *create_instance(const char *md, const char *jd) {
    (void)md;(void)jd;
    wurl_t *w=(wurl_t*)calloc(1,sizeof(wurl_t)); if(!w) return NULL;
    w->p_vol=0.5f; w->p_trem=0.5f; w->p_spk=0.5f; w->p_atk=0.5f;
    w->p_dcy=0.5f; w->p_brt=0.5f; w->p_bark=0.5f; w->p_tune=0.5f;
    w->s_vol=w->p_vol; w->s_trem=w->p_trem; w->s_spk=w->p_spk;
    preamp_init(&w->preamp,SAMPLE_RATE_D);
    tremolo_init(&w->tremolo,0.5,SAMPLE_RATE_D);
    pa_init(&w->pa); spk_init(&w->speaker,SAMPLE_RATE_D);
    return w;
}
static void destroy_instance(void *inst) { free(inst); }

/* ── MIDI ────────────────────────────────────────────────────────── */
static void on_midi(void *instance, const uint8_t *msg, int len, int src) {
    (void)src; wurl_t *w=(wurl_t*)instance; if(len<3) return;
    uint8_t st=msg[0]&0xF0, d1=msg[1], d2=msg[2];
    if (st==0x90 && d2>0) {
        int slot=-1; uint64_t oldest=UINT64_MAX;
        for (int i=0;i<NUM_VOICES;i++) if(!w->voices[i].active){slot=i;break;}
        if (slot<0) { uint64_t or2=UINT64_MAX;
            for(int i=0;i<NUM_VOICES;i++) if(w->voices[i].releasing&&w->voices[i].age<or2){or2=w->voices[i].age;slot=i;}
        }
        if (slot<0) for(int i=0;i<NUM_VOICES;i++) if(w->voices[i].age<oldest){oldest=w->voices[i].age;slot=i;}
        if (slot<0) slot=0;
        double vel=d2/127.0, tune=((double)w->p_tune-0.5)*200.0+(double)w->pitch_bend*200.0;
        uint32_t seed=((uint32_t)d1*2654435761u)^(uint32_t)w->age_counter;
        voice_note_on(&w->voices[slot],d1,vel,SAMPLE_RATE_D,seed,
            (double)w->p_brt,(double)w->p_bark,(double)w->p_atk,(double)w->p_dcy,tune);
        w->voices[slot].age=w->age_counter++;
    } else if (st==0x80||(st==0x90&&d2==0)) {
        for(int i=0;i<NUM_VOICES;i++)
            if(w->voices[i].active&&!w->voices[i].releasing&&w->voices[i].midi_note==d1)
                {voice_note_off(&w->voices[i]);break;}
    } else if (st==0xE0) {
        w->pitch_bend=(float)((d2<<7|d1)-8192)/8192.0f;
    }
}

/* ── Params ──────────────────────────────────────────────────────── */
static float *param_ptr(wurl_t *w, int i) {
    float *p[]={&w->p_vol,&w->p_trem,&w->p_spk,&w->p_atk,&w->p_dcy,&w->p_brt,&w->p_bark,&w->p_tune};
    return (i>=0&&i<8)?p[i]:NULL;
}

static void set_param(void *instance, const char *key, const char *val) {
    wurl_t *w=(wurl_t*)instance; if(!w||!key||!val) return;
    if(strncmp(key,"knob_",5)==0&&strstr(key,"_adjust")) {
        int idx=atoi(key+5)-1; float *p=param_ptr(w,idx);
        if(p){int d=atoi(val);*p=clampf(*p+d*KNOBS[idx].step,KNOBS[idx].min,KNOBS[idx].max);}
        return;
    }
    float f=(float)atof(val);
    if(!strcmp(key,"volume"))w->p_vol=clampf(f,0,1);
    else if(!strcmp(key,"tremolo"))w->p_trem=clampf(f,0,1);
    else if(!strcmp(key,"speaker"))w->p_spk=clampf(f,0,1);
    else if(!strcmp(key,"attack"))w->p_atk=clampf(f,0,1);
    else if(!strcmp(key,"decay"))w->p_dcy=clampf(f,0,1);
    else if(!strcmp(key,"brightness"))w->p_brt=clampf(f,0,1);
    else if(!strcmp(key,"bark"))w->p_bark=clampf(f,0,1);
    else if(!strcmp(key,"tune"))w->p_tune=clampf(f,0,1);
    else if(!strcmp(key,"state"))
        sscanf(val,"%f %f %f %f %f %f %f %f",&w->p_vol,&w->p_trem,&w->p_spk,&w->p_atk,&w->p_dcy,&w->p_brt,&w->p_bark,&w->p_tune);
}

static int get_param(void *instance, const char *key, char *buf, int bl) {
    wurl_t *w=(wurl_t*)instance;
    if(!strcmp(key,"chain_params")) return snprintf(buf,bl,
        "[{\"key\":\"volume\",\"name\":\"Volume\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
        "{\"key\":\"tremolo\",\"name\":\"Tremolo\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
        "{\"key\":\"speaker\",\"name\":\"Speaker\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
        "{\"key\":\"attack\",\"name\":\"Attack\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
        "{\"key\":\"decay\",\"name\":\"Decay\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
        "{\"key\":\"brightness\",\"name\":\"Bright\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
        "{\"key\":\"bark\",\"name\":\"Bark\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01},"
        "{\"key\":\"tune\",\"name\":\"Tune\",\"type\":\"float\",\"min\":0,\"max\":1,\"step\":0.01}]");
    if(!strcmp(key,"ui_hierarchy")) {
        static const char ui[]="{\"modes\":null,\"levels\":{\"root\":{\"name\":\"Wurl\","
            "\"knobs\":[\"volume\",\"tremolo\",\"speaker\",\"attack\",\"decay\",\"brightness\",\"bark\",\"tune\"],"
            "\"params\":[\"volume\",\"tremolo\",\"speaker\",\"attack\",\"decay\",\"brightness\",\"bark\",\"tune\"]}}}";
        int l=(int)strlen(ui); if(l<bl) memcpy(buf,ui,l+1); return l;
    }
    if(strncmp(key,"knob_",5)==0&&strstr(key,"_name")){int i=atoi(key+5)-1;if(i>=0&&i<8)return snprintf(buf,bl,"%s",KNOBS[i].label);return -1;}
    if(strncmp(key,"knob_",5)==0&&strstr(key,"_value")){int i=atoi(key+5)-1;float*p=param_ptr(w,i);if(p)return snprintf(buf,bl,"%d%%",(int)(*p*100));return -1;}
    if(!strcmp(key,"name")) return snprintf(buf,bl,"Wurl");
    if(!strcmp(key,"volume"))return snprintf(buf,bl,"%.4f",w->p_vol);
    if(!strcmp(key,"tremolo"))return snprintf(buf,bl,"%.4f",w->p_trem);
    if(!strcmp(key,"speaker"))return snprintf(buf,bl,"%.4f",w->p_spk);
    if(!strcmp(key,"attack"))return snprintf(buf,bl,"%.4f",w->p_atk);
    if(!strcmp(key,"decay"))return snprintf(buf,bl,"%.4f",w->p_dcy);
    if(!strcmp(key,"brightness"))return snprintf(buf,bl,"%.4f",w->p_brt);
    if(!strcmp(key,"bark"))return snprintf(buf,bl,"%.4f",w->p_bark);
    if(!strcmp(key,"tune"))return snprintf(buf,bl,"%.4f",w->p_tune);
    if(!strcmp(key,"state"))return snprintf(buf,bl,"%f %f %f %f %f %f %f %f",
        w->p_vol,w->p_trem,w->p_spk,w->p_atk,w->p_dcy,w->p_brt,w->p_bark,w->p_tune);
    return -1;
}

/* ── Render ──────────────────────────────────────────────────────── */
static void render_block(void *instance, int16_t *out_lr, int frames) {
    wurl_t *w=(wurl_t*)instance;
    if (frames > BLOCK_SIZE) frames = BLOCK_SIZE;
    #define SM 0.25f
    w->s_vol+=SM*(w->p_vol-w->s_vol);
    w->s_trem+=SM*(w->p_trem-w->s_trem);
    w->s_spk+=SM*(w->p_spk-w->s_spk);
    tremolo_set_depth(&w->tremolo,(double)w->s_trem);
    spk_set_char(&w->speaker,(double)w->s_spk,SAMPLE_RATE_D);
    memset(w->sbuf,0,frames*sizeof(double));
    for (int v=0;v<NUM_VOICES;v++) {
        if(!w->voices[v].active) continue;
        voice_render(&w->voices[v],w->vbuf,frames);
        for(int i=0;i<frames;i++) w->sbuf[i]+=w->vbuf[i];
        if(w->voices[v].releasing&&reed_is_silent(&w->voices[v].reed))
            w->voices[v].active=0;
    }
    double vol=(double)w->s_vol; vol*=vol;
    for (int i=0;i<frames;i++) {
        double x=w->sbuf[i];
        preamp_set_ldr(&w->preamp,tremolo_process(&w->tremolo));
        x=preamp_process(&w->preamp,x);
        x*=vol;
        x=pa_process(&w->pa,x);
        x=spk_process(&w->speaker,x);
        x*=POST_SPEAKER_GAIN;
        x=tanh(x*1.2)*0.85;
        int16_t s=(int16_t)(clamp_d(x,-1.0,1.0)*32767.0);
        out_lr[i*2]=s; out_lr[i*2+1]=s;
    }
}

/* ── API v2 ──────────────────────────────────────────────────────── */
typedef struct {
    uint32_t api_version;
    void*(*create_instance)(const char*,const char*);
    void(*destroy_instance)(void*);
    void(*on_midi)(void*,const uint8_t*,int,int);
    void(*set_param)(void*,const char*,const char*);
    int(*get_param)(void*,const char*,char*,int);
    int(*get_error)(void*,char*,int);
    void(*render_block)(void*,int16_t*,int);
} plugin_api_v2_t;

__attribute__((visibility("default")))
plugin_api_v2_t* move_plugin_init_v2(const void *host) {
    (void)host;
    static plugin_api_v2_t api={
        .api_version=2,.create_instance=create_instance,.destroy_instance=destroy_instance,
        .on_midi=on_midi,.set_param=set_param,.get_param=get_param,
        .get_error=NULL,.render_block=render_block,
    };
    return &api;
}
