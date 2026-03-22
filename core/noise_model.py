"""
Noise model and link budget for orbital phase-rate communication.

Models the realistic noise floor that seed-encoded ΔV signals must
overcome: thermal noise, clock jitter, ionospheric scintillation,
unmodeled perturbation residuals, and quantization noise.
"""

import numpy as np
from .physics_constants import (
    MU_EARTH, R_EARTH, ALTITUDE_REF, LAMBDA_RF, C,
    J2_EARTH, R_EARTH_EQUATORIAL, P_SOLAR,
    SAT_MASS_DEFAULT, SAT_AREA_DEFAULT, SAT_CR_DEFAULT
)


class PhaseRateNoiseModel:
    """
    Models all noise sources that corrupt phase-rate observations.

    Noise sources (all converted to rad/s in phase-rate domain):
        1. Thermal receiver noise (white, Gaussian)
        2. Clock instability (Allan deviation → random walk)
        3. Ionospheric scintillation (correlated, 1/f-like)
        4. Unmodeled perturbation residuals (SRP, drag, higher gravity)
        5. Quantization noise (ADC resolution)
    """

    def __init__(self, lambda_rf=LAMBDA_RF, altitude=ALTITUDE_REF,
                 receiver_bandwidth=1.0, carrier_snr_db=30.0,
                 clock_allan_dev=1e-11, iono_strength=1e-9,
                 sample_rate=1.0):
        """
        Initialize noise model.

        Args:
            lambda_rf: RF wavelength in meters
            altitude: Orbital altitude in meters
            receiver_bandwidth: Receiver bandwidth in Hz
            carrier_snr_db: Carrier signal-to-noise ratio in dB
            clock_allan_dev: Allan deviation of oscillator at 1s
            iono_strength: Ionospheric phase rate noise amplitude (rad/s)
            sample_rate: Sample rate in Hz
        """
        self.lambda_rf = lambda_rf
        self.altitude = altitude
        self.bandwidth = receiver_bandwidth
        self.carrier_snr = 10 ** (carrier_snr_db / 10)
        self.clock_allan = clock_allan_dev
        self.iono_strength = iono_strength
        self.sample_rate = sample_rate

        # Derived parameters
        self.range_typical = self._typical_range()

    def _typical_range(self):
        """Estimate typical inter-satellite range for link budget."""
        a = R_EARTH + self.altitude
        return 2 * a * np.sin(np.pi / 6)  # ~60 deg separation

    def thermal_noise_std(self):
        """
        Thermal phase-rate noise standard deviation.

        From Cramer-Rao bound on phase-rate estimation:
            sigma_dphi = (1 / SNR) * sqrt(12 * B^3 / N) * (2*pi/lambda)

        For range-rate: sigma_rdot = lambda / (2*pi) * sigma_dphi
        Simplified: sigma_dphi ~ (2*pi/lambda) * (c / (4*pi*f*SNR)) * sqrt(B)

        Returns:
            Standard deviation in rad/s
        """
        sigma_range_rate = C / (4 * np.pi * (C / self.lambda_rf) *
                                np.sqrt(self.carrier_snr))
        sigma_phase_rate = (2 * np.pi / self.lambda_rf) * sigma_range_rate
        return sigma_phase_rate * np.sqrt(self.bandwidth)

    def clock_noise_std(self, tau):
        """
        Clock instability contribution to phase-rate noise.

        Allan deviation scales as: sigma_y(tau) = allan_dev / sqrt(tau)
        Phase rate noise: sigma_dphi = (2*pi*f) * sigma_y

        Args:
            tau: Integration time in seconds

        Returns:
            Standard deviation in rad/s
        """
        freq = C / self.lambda_rf
        sigma_y = self.clock_allan / np.sqrt(max(tau, 0.01))
        return 2 * np.pi * freq * sigma_y

    def iono_noise_psd(self, freqs):
        """
        Ionospheric scintillation power spectral density.

        1/f spectrum with high-frequency cutoff at ~1 Hz.

        Args:
            freqs: Frequency array in Hz

        Returns:
            PSD array in (rad/s)^2/Hz
        """
        f_cutoff = 1.0
        psd = self.iono_strength**2 / (np.abs(freqs) + 0.01)
        psd *= 1.0 / (1.0 + (freqs / f_cutoff)**2)
        return psd

    def perturbation_residual_std(self):
        """
        Unmodeled perturbation noise floor.

        Even with J2 in the propagator, residuals from SRP, drag,
        higher-order gravity, and Earth tides create a noise floor.

        Returns:
            Standard deviation of phase-rate residual in rad/s
        """
        a = R_EARTH + self.altitude
        v_orb = np.sqrt(MU_EARTH / a)

        # SRP acceleration residual (~10% modeling error)
        a_srp = P_SOLAR * SAT_CR_DEFAULT * SAT_AREA_DEFAULT / SAT_MASS_DEFAULT
        srp_residual = 0.1 * a_srp

        # Higher-order gravity (J3, J4, tesseral harmonics)
        # J3 ~ 2.5e-6, effect ~ J3/J2 * J2_effect
        j3_residual = 2.5e-6 / J2_EARTH * 1.5 * J2_EARTH * MU_EARTH * \
            R_EARTH_EQUATORIAL**2 / a**4

        # Total acceleration residual → range-rate noise
        total_acc_residual = np.sqrt(srp_residual**2 + j3_residual**2)

        # Convert acceleration to phase-rate: dphi/dt ~ (2*pi/lambda) * (a*T)
        # where T is observation time (~symbol_period)
        symbol_period = 300.0
        sigma_range_rate = total_acc_residual * symbol_period
        sigma_phase_rate = (2 * np.pi / self.lambda_rf) * sigma_range_rate

        return sigma_phase_rate

    def total_noise_std(self, integration_time=300.0):
        """
        Combined noise standard deviation from all sources.

        Args:
            integration_time: Observation integration time in seconds

        Returns:
            Total noise standard deviation in rad/s
        """
        sigma_thermal = self.thermal_noise_std()
        sigma_clock = self.clock_noise_std(integration_time)
        sigma_perturb = self.perturbation_residual_std()
        sigma_iono = self.iono_strength

        total = np.sqrt(sigma_thermal**2 + sigma_clock**2 +
                        sigma_perturb**2 + sigma_iono**2)
        return total

    def generate_noise(self, n_samples, integration_time=300.0, rng=None):
        """
        Generate realistic noise time series.

        Combines white thermal noise with colored ionospheric noise.

        Args:
            n_samples: Number of samples
            integration_time: Observation window in seconds
            rng: NumPy random generator (or None for default)

        Returns:
            noise: Array of phase-rate noise samples (rad/s)
        """
        if rng is None:
            rng = np.random.default_rng()

        # White noise (thermal + clock + perturbation)
        sigma_white = np.sqrt(
            self.thermal_noise_std()**2 +
            self.clock_noise_std(integration_time)**2 +
            self.perturbation_residual_std()**2
        )
        white = rng.normal(0, sigma_white, n_samples)

        # Colored noise (ionospheric scintillation via spectral shaping)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.sample_rate)
        psd = self.iono_noise_psd(freqs)
        amplitudes = np.sqrt(psd * self.sample_rate / 2)
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        spectrum = amplitudes * np.exp(1j * phases)
        colored = np.fft.irfft(spectrum, n=n_samples)

        return white + colored

    def link_budget(self, delta_v=0.001, symbol_period=300.0):
        """
        Compute link budget: signal strength vs noise floor.

        Args:
            delta_v: Applied ΔV magnitude in m/s
            symbol_period: Symbol period in seconds

        Returns:
            Dictionary with link budget parameters
        """
        a = R_EARTH + self.altitude
        v_orb = np.sqrt(MU_EARTH / a)

        # Signal: phase-rate deviation from ΔV
        # ΔV changes range rate by ~ΔV (direct, for along-track impulse)
        signal_range_rate = delta_v
        signal_phase_rate = (2 * np.pi / self.lambda_rf) * signal_range_rate

        # Noise floor
        sigma_total = self.total_noise_std(symbol_period)

        # SNR
        snr_linear = signal_phase_rate / sigma_total
        snr_db = 20 * np.log10(max(snr_linear, 1e-20))

        # Integration gain (averaging over symbol period)
        n_samples = int(symbol_period * self.sample_rate)
        integration_gain_db = 10 * np.log10(max(n_samples, 1))

        # Effective SNR after integration
        effective_snr_db = snr_db + integration_gain_db

        return {
            'delta_v_ms': delta_v,
            'signal_phase_rate': signal_phase_rate,
            'noise_std': sigma_total,
            'snr_single_sample_db': snr_db,
            'integration_gain_db': integration_gain_db,
            'effective_snr_db': effective_snr_db,
            'n_samples': n_samples,
            'thermal_noise': self.thermal_noise_std(),
            'clock_noise': self.clock_noise_std(symbol_period),
            'perturbation_noise': self.perturbation_residual_std(),
            'iono_noise': self.iono_strength,
        }


def quick_test():
    """Quick sanity check."""
    print("Testing PhaseRateNoiseModel...")

    model = PhaseRateNoiseModel()

    # Link budget for 1 mm/s ΔV
    budget = model.link_budget(delta_v=0.001, symbol_period=300.0)

    print(f"  Signal phase rate: {budget['signal_phase_rate']:.4f} rad/s")
    print(f"  Noise std:         {budget['noise_std']:.6f} rad/s")
    print(f"  SNR (single):      {budget['snr_single_sample_db']:.1f} dB")
    print(f"  Integration gain:  {budget['integration_gain_db']:.1f} dB")
    print(f"  Effective SNR:     {budget['effective_snr_db']:.1f} dB")
    print(f"  Noise breakdown:")
    print(f"    Thermal:     {budget['thermal_noise']:.2e} rad/s")
    print(f"    Clock:       {budget['clock_noise']:.2e} rad/s")
    print(f"    Perturbation:{budget['perturbation_noise']:.2e} rad/s")
    print(f"    Ionospheric: {budget['iono_noise']:.2e} rad/s")

    # Generate noise
    noise = model.generate_noise(300)
    print(f"  Generated noise: {len(noise)} samples, std={np.std(noise):.2e} rad/s")

    print("Test passed!")


if __name__ == '__main__':
    quick_test()
