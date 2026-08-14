# Remote Sensing and Geospatial Machine Learning

Satellites are the only instrument we have that observes the whole planet, repeatedly, with a
consistent measurement. That combination — global coverage, regular revisit, uniform
methodology — is what makes remote sensing central to climate and environmental science, and
it is also what makes it a natural fit for machine learning. A single Sentinel-2 scene contains
more pixels than a person could inspect in a lifetime, and the archive grows by terabytes a day.

## What a satellite actually measures

It is worth being precise about this, because it shapes everything downstream. A passive optical
sensor measures **radiance** — the energy arriving at the detector in a set of wavelength bands.
After correcting for atmospheric effects, this is converted to **surface reflectance**: the
fraction of incoming sunlight the surface sends back, in each band.

So a satellite image is not a photograph. It is a stack of physical measurements, one per
**spectral band**, and each pixel is a vector of reflectance values. Sentinel-2 measures 13
bands; Landsat 8/9 measures 11. Working with this data means working with that spectral
dimension, not just spatial structure.

Different surfaces reflect differently across wavelengths, and those differences are the entire
basis for what follows:

- **Vegetation** absorbs strongly in the red (chlorophyll) and reflects strongly in the
  near-infrared (leaf cell structure). The resulting red/NIR contrast is the single most useful
  signal in optical remote sensing.
- **Water** absorbs nearly everything beyond the visible, so it is very dark in NIR and SWIR.
- **Bare soil** reflects moderately and increases smoothly with wavelength.
- **Built surfaces** tend to be spectrally flat and bright, which is why they are easy to
  confuse with soil.

## The four resolutions

Every sensor represents a trade-off, and no satellite is good at everything:

- **Spatial** — how large an area one pixel covers (Sentinel-2: 10–60 m; MODIS: 250–1000 m)
- **Spectral** — how many bands, and how narrow (multispectral: ~10 bands;
  hyperspectral: hundreds)
- **Temporal** — how often the sensor revisits (Sentinel-2: ~5 days; geostationary: minutes)
- **Radiometric** — how finely the sensor resolves brightness differences

The trade-off is physical, not technological. Finer spatial resolution means fewer photons per
pixel, which must be paid for with a wider band or a longer dwell — so higher spatial resolution
generally costs spectral resolution or revisit frequency. Choosing a sensor means deciding which
of these your problem actually needs. Mapping individual buildings needs spatial resolution;
tracking a plume needs revisit frequency; identifying a gas needs spectral resolution.

## Why machine learning

Three properties of this data make it a good match for ML:

**Volume.** The Sentinel and Landsat archives are petabyte-scale and open. There is far more
data than can be analyzed by hand, and the interesting signals are often rare.

**High-dimensional, correlated features.** Each pixel is a spectral vector, and neighboring
pixels carry spatial context. Both classical methods (random forests on spectral features) and
deep learning (CNNs exploiting spatial structure) have natural roles.

**Labels are the bottleneck.** Imagery is abundant; ground truth is expensive. Much of the
methodological work in this field is about learning effectively from limited, unevenly
distributed labels — which is also why validation is so easy to get wrong here.

## Applications in climate and environmental work

- **Land cover and land use change** — deforestation, agricultural expansion, urban growth.
  Land use change is a major term in the carbon budget, and satellites are how it is quantified.
- **Emissions detection** — methane plumes from oil and gas infrastructure and landfills, using
  instruments like TROPOMI, EMIT, and commercial hyperspectral sensors. Methane has an outsized
  near-term warming effect and leaks are often unreported, so detection has direct mitigation value.
- **Biomass and carbon stocks** — estimating above-ground carbon from optical, radar, and lidar
- **Surface properties for climate models** — albedo, land surface temperature, snow and ice extent
- **Disaster response and attribution** — flood extent, burn scars, drought monitoring

## A warning that carries over from the last unit

Remote sensing data is the clearest case of everything covered in **Model Evaluation and
Validation**. Pixels are strongly spatially autocorrelated — adjacent pixels usually belong to
the same land cover patch and often to the same physical object. A random train/test split over
pixels puts a pixel's immediate neighbors into the training set, and the model scores well by
recognizing that specific patch rather than by learning what vegetation looks like.

Published land cover accuracies have been inflated this way often enough that it is a known
methodological problem in the literature. The fix is the same as before: hold out contiguous
spatial blocks, or better, hold out entirely separate scenes or regions. The notebook that
follows measures the size of this effect on a land cover problem.
