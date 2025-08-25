<p align="center">
   <img src="assets/logo-suprasense.svg" alt="Logo" style="width:80%; display:block; margin:auto;">
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build and Release](https://github.com/ahmadomira/fitting-tool/actions/workflows/build_and_release.yml/badge.svg)](https://github.com/ahmadomira/fitting-tool/actions/workflows/build_and_release.yml)
[![Latest Release](https://img.shields.io/github/v/release/ahmadomira/fitting-tool?label=Latest%20Release)](https://github.com/ahmadomira/fitting-tool/releases/latest)

# Molecular Binding Assay Fitting Toolkit

A Python-based application for analyzing molecular binding interactions using various spectroscopic assays. This tool provides robust mathematical fitting algorithms based on **forward modeling** approaches that overcome limitations of traditional data transformation methods commonly used in supramolecular chemistry and biochemistry.

## 🧬 Overview

### Forward Modeling Approach

Unlike traditional binding analysis methods that rely on data transformation (Scatchard plots, Hill plots, double-reciprocal plots), this application employs **direct forward modeling** where theoretical binding curves are fitted directly to raw experimental data. This approach offers several key advantages:

- **Preserves Error Structure**: Maintains native measurement uncertainties without distortion from data transformation
- **Physical Constraints**: Parameters are bounded by thermodynamically reasonable values, ensuring physically meaningful results
- **Handles Complex Systems**: Naturally accommodates competitive binding and ligand depletion scenarios
- **Robust Validation**: Model assessment occurs in the original measurement space where experimental errors are best understood

### Supported Assay Types

This application enables analysis of experimental data from various spectroscopic assays:

- **GDA (Guest Displacement Assay)**: Quantifies guest binding affinity by monitoring displacement of a preformed host-indicator complex - particularly valuable for spectroscopically silent hosts and guests, and superior for insoluble or weakly binding guests
- **IDA (Indicator Displacement Assay)**: Determines binding constants using competitive displacement of indicator dyes - enables detection in complex biological matrices through ultra-high-affinity reporter pairs
- **DBA (Direct Binding Assay)**: Direct measurement of host-guest or dye-host binding interactions available in both titration modes - monitors spectroscopic changes upon complex formation but limited in complex matrices due to competitive binding from naturally occurring interferents
- **Dye Alone**: Linear calibration fitting for indicator dyes - establishes baseline fluorescence properties and corrects for inner filter effects in competitive binding assays

The application features an intuitive Tkinter-based graphical user interface with advanced data visualization, comprehensive statistical analysis, and flexible result export capabilities. 

## 🚀 Features

### Core Analysis Capabilities
- **Multiple Assay Types**: Support for GDA, IDA, DBA (both host-to-dye and dye-to-host titrations), and dye alone calibration
- **Advanced Optimization**: L-BFGS-B constrained optimization with multi-start global optimization for robust parameter estimation
- **Ensemble-Based Uncertainty**: Parameter uncertainties estimated through ensemble statistics from multiple optimization runs, capturing nonlinear uncertainty propagation
- **Physical Constraints**: Thermodynamically bounded parameters ensure meaningful binding constants and fluorescence coefficients

### Statistical Analysis & Quality Control
- **Multi-Level Quality Assessment**: R²/RMSE filtering with adaptive thresholds and parameter ensemble validation
- **Robust Parameter Estimation**: Median-based statistics from optimization ensembles provide outlier-resistant results
- **Comprehensive Error Analysis**: Confidence intervals that account for both optimization uncertainty and experimental variability
- **Model Validation**: Goodness-of-fit metrics and residual analysis in the original measurement space

### Data Processing & Visualization
- **Data Visualization**: Publication-quality plots with unified styling and export options
- **Result Merging**: Statistical combination of multiple fitting results with outlier detection using median absolute deviation criteria
- **Cross-Platform**: Available for Windows, macOS, and Linux with automated build system
- **Laboratory Integration**: Direct processing of BMG plate reader files with standardized data workflows

### Planned Enhancements
- 🚧 **Parallel Processing**: Multi-threading support for simultaneous analysis of multiple datasets
- 🚧 **Automated Testing**: Comprehensive validation of mathematical algorithms and data processing
- 🚧 **Jupyter Integration**: Included notebooks for advanced analysis and method development

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Operating System: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- RAM: 2GB minimum, 4GB recommended
- Storage: 500MB free space

### Python Dependencies

**Core Analysis Engine:**
```
matplotlib>=3.5.0  # Publication-quality visualizations with customizable styling
numpy>=1.21.0      # Numerical array operations and mathematical computations
pandas>=1.3.0      # BMG Labtech plate reader file processing and structured data manipulation  
scipy>=1.7.0       # L-BFGS-B optimization and statistical functions
```

**Key Technology Choices:**
- **scipy.optimize**: L-BFGS-B constrained optimization selected for superior handling of parameter bounds with active-set strategy
- **Tkinter**: Cross-platform GUI compatibility without external dependencies
- **Cross-Platform Distribution**: PyInstaller-based executable generation through GitHub Actions CI/CD
- **BMG Labtech Plate Readers**: Direct processing of plate reader Excel exports with automatic 96-well layout extraction for high-throughput analysis

## 🛠️ Installation

### Option 1: Pre-built Executables (Recommended for End Users)

Download the latest release for your operating system from the [Releases](../../releases) page:

- **Windows**: `windows-executable.zip`
- **macOS**: `macos-executable.zip`
- **Linux** (Ubuntu): `linux-executable.zip`

### Option 2: Run from Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fitting_app.git
   cd fitting_app
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python main.py
   ```

### Option 3: Development Installation

For developers wanting to contribute or modify the code:

```bash
git clone https://github.com/your-username/fitting_app.git
cd fitting_app
pip install -e .
pip install -r requirements.txt
```

## 📖 Usage Guide

### Getting Started

1. **Launch the application** by using the pre-built executable or running `python main.py` 
2. **Select your assay type** from the main interface
3. **Load your data** using the file browser (supports .txt files, see `./data`)
4. **Configure fitting parameters** in the interface
5. **Run the fitting** and analyze results

### Data Format

Currently, input data should be in tab-delimited format with the following structure:

#### For DBA/GDA/IDA Assays:
```
var   signal
0.0   1.000
1e-6  0.995
2e-6  0.987
...
```

### 🚧 Advanced Features

#### Result Merging
Combine multiple independent fitting results and perform a final fit:
1. Select "Merge Fits" for your assay type
2. Load multiple result files
3. Perform a final fit over combined fits
3. Generate combined statistics and plots

## 📁 Project Structure

```
fitting_app/
├── main.py                   # Application entry point
├── requirements.txt
├── README.md                 # This file
├── core/                     # Core fitting algorithms
│   ├── fitting/              # Assay-specific fitting modules
│   │   ├── gda.py 
│   │   ├── ida.py 
│   │   ├── dba_host_to_dye.py
│   │   ├── dba_dye_to_host.py 
│   │   └── dye_alone.py 
│   ├── base_fitting_app.py    # Base fitting framework
│   ├── fitting_utils.py       # Utility functions
│   └── forward_model.py       # Mathematical models
├── gui/                       # User interface modules
│   ├── main_interface.py
│   ├── interface_*_fitting.py
│   └── interface_*_merge_fits.py
├── utils/                     # General utilities
├── assets/                    # icons & logos
├── tests/                     # Unit tests
├── examples/                  # Example data and scripts
├── notebooks/                 # Jupyter analysis notebooks
├── data/                      # Sample datasets
├── docs/
└── build/
    └── FittingApp.spec        # Build configuration
```

## 🔬 Scientific Background

### Binding Models

The application implements rigorous binding models based on mass action kinetics and equilibrium thermodynamics:

#### Direct Binding Assay (DBA)

DBA measures direct binding interactions through spectroscopic signal changes upon complex formation.

**Equilibrium:**
```
Host + Dye ⇌ Host-Dye Complex
Ka = [Host-Dye] / ([Host][Dye])
```

**Signal Equation:**
```
I = I₀ + I_dye × [Dye] + I_complex × [Host-Dye]
```

Where I₀ is baseline fluorescence, and I_dye, I_complex are molar fluorescence coefficients.

#### Competitive Binding Assays (IDA & GDA)

These assays involve coupled equilibria where two species compete for the same binding site.

**Equilibria:**
```
Host + Dye ⇌ Host-Dye     (Ka_dye)
Host + Guest ⇌ Host-Guest (Ka_guest)  
```

The mathematical framework handles the coupled mass balance equations numerically, accounting for competitive displacement based on relative binding affinities.

### Key Methodological Advances

- **Forward Modeling**: Direct fitting of binding equations to raw data, avoiding transformation artifacts
- **Physical Constraints**: Binding constants bounded by thermodynamically reasonable ranges (10² to 10¹⁰ M⁻¹)
- **Statistical Rigor**: Multiple optimization runs provide robust uncertainty estimates for all parameters

## 🔧 Building from Source & Quality Assurance

### Prerequisites for Building
- Python 3.8+
- PyInstaller
- All dependencies from requirements.txt

### Build Commands

**Build executable:**
```bash
pyinstaller --clean -y --distpath ./dist --workpath ./build FittingApp.spec
```

**Development build:**
```bash
python -m build
```

The built executable will be available in the `dist/` directory.

## 🧪 Testing & Validation

**Current Validation Methods:**
- **Real-world Performance**: Typical fitting quality metrics (R² > 0.99) across diverse molecular systems
- **Parameter Consistency**: Multi-start optimization ensures reproducible binding constants within statistical uncertainty
- **Physical Validation**: All fitted parameters maintain thermodynamic validity and experimental constraints

**Future Testing Framework:**
- 🚧 Comprehensive unittest suite for core algorithms
- 🚧 Integration tests for data processing workflows  
- 🚧 Performance benchmarking across dataset sizes
- 🚧 Cross-validation with literature binding constants 

### Quality Assurance

- **Version Control**: Git-based change tracking and collaboration with systematic commit practices
- **Continuous Deployment**: Automated multi-platform builds (Windows, macOS, Linux) through GitHub Actions CI/CD pipeline
- **Mathematical Validation**: Forward modeling algorithms validated against theoretical binding isotherms

**Planned Quality Enhancements:**
- 🚧 **Cross-Platform Testing**: Automated validation across operating systems ensures consistent performance
- 🚧 **Continuous Integration**: Enhanced cross-platform compatibility verification with automated regression testing
- 🚧 **Code Standardization**: Automated formatting (Black) and style checking (pylint) for maintainable codebase

## 🤝 Contributing

We welcome contributions! Our contributing guideline:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Ensure tests pass:** `python -m pytest`
5. **Commit changes:** `git commit -m 'Add amazing feature'`
6. **Push to branch:** `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### 🚧 Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes following [Numpy's docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
- Include unit tests for new features

## 📊 Example Workflows

### Basic IDA Analysis

🚧 Coming soon

### Batch Processing

🚧 Coming soon

## 📈 Performance

- **Fitting Speed**: Typical datasets (8 replicas X 11 measurement points) fit in <1 second
- **Memory Usage**: <100MB for standard datasets
- **Scalability**: Tested with datasets up to 10,000 points

## 📝 Citation

🚧 WIP

## 🐛 Troubleshooting

**Fitting convergence issues:**
- Check data quality and remove outliers
- Adjust initial parameter estimates
- Verify data format matches expected structure

### Getting Help

- **Issues**: Report bugs via [GitHub Issues](https://github.com/ahmadomira/fitting-tool/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/ahmadomira/fitting-tool/discussions)
- **Documentation Page**: 🚧 Coming soon

## 📄 License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

🚧 WIP

## References

1. Sinn, S., Spuling, E., Bräse, S., & Biedermann, F. (2019). Rational design and implementation of a cucurbit[8]uril-based indicator-displacement assay for application in blood serum. *Chemical Science*, 10(28), 6584-6593. https://doi.org/10.1039/C9SC00705A

2. Sinn, S., Krämer, J., & Biedermann, F. (2020). Teaching old indicators even more tricks: binding affinity measurements with the guest-displacement assay (GDA). *Chemical Communications*, 56(49), 6620-6623. https://doi.org/10.1039/D0CC01841D

---

**Last Updated**: August 2025

**Contact:** contact@suprabank.org
