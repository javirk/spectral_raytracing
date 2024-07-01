import numpy as np

min_wavelength = 450
max_wavelength = 750
num_wavelengths = 12

wavelengths = np.linspace(min_wavelength, max_wavelength, num_wavelengths)

def xfit_1931(wave):
    t1 = (wave - 442.0) * (0.0624 if wave < 442.0 else 0.0374)
    t2 = (wave - 599.8) * (0.0264 if wave < 599.8 else 0.0323)
    t3 = (wave - 501.1) * (0.0490 if wave < 501.1 else 0.0382)
        
    return 0.362 * np.exp(-0.5 * t1**2) + 1.056 * np.exp(-0.5 * t2**2) - 0.065 * np.exp(-0.5 * t3**2)

def yfit_1931(wave):
    t1 = (wave - 568.8) * (0.0213 if wave < 568.8 else 0.0247)
    t2 = (wave - 530.9) * (0.0613 if wave < 530.9 else 0.0322)
    return 0.821 * np.exp(-0.5 * t1**2) + 0.286 * np.exp(-0.5 * t2**2)

def zfit_1931(wave):
    t1 = (wave - 437.0) * (0.0845 if wave < 437.0 else 0.0278)
    t2 = (wave - 459.0) * (0.0385 if wave < 459.0 else 0.0725)
    return 1.217 * np.exp(-0.5 * t1**2) + 0.681 * np.exp(-0.5 * t2**2)

def a_illuminant(wave):
    den = np.exp(1.435e7 / (2848. * wave)) - 1.
    num = np.exp(1.435e7 / (2848. * 560)) - 1.

    return 100. * np.power(560. / wave, 5.) * num / den

def xyz_to_rgb(x, y, z):
    r = x * 3.2409699 + y * -1.5373832 + z * -0.4986108
    g = x * -0.9692436 + y * 1.8759675 + z * 0.0415551
    b = x * 0.0556301 + y * -0.2039770 + z * 1.0569715

    return r, g, b

def calculate_normalization_factor():
    normalization_factor = 0.
    for wave in wavelengths:
        normalization_factor += yfit_1931(wave)
    return normalization_factor

def main():
    x, y, z = 0, 0, 0
    normalization_factor = calculate_normalization_factor()

    for wave in wavelengths:
        ill = a_illuminant(wave)
        x += xfit_1931(wave) * ill
        y += yfit_1931(wave) * ill
        z += zfit_1931(wave) * ill

    x /= normalization_factor
    y /= normalization_factor
    z /= normalization_factor

    total = x + y + z

    x /= total
    y /= total
    z = 1 - x - y
    r, g, b = xyz_to_rgb(x, y, z)


    print(f"RGB: {r}, {g}, {b}")


if __name__ == "__main__":
    main()