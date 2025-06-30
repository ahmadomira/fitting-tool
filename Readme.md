[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build and Release](https://github.com/ahmadomira/fitting-tool/actions/workflows/build_and_release.yml/badge.svg)](https://github.com/ahmadomira/fitting-tool/actions/workflows/build_and_release.yml)

# Molecular Binding Assay Fitting Toolkit

A Python-based application for analyzing molecular binding interactions using various spectroscopic assays. This tool provides robust mathematical fitting algorithms for different types of binding experiments commonly used in supramolecular chemistry and biochemistry. 

## 🧬 Overview

This application enables analysis of experimental data from various spectroscopic assays:

- **GDA (Guest Displacement Assay)**: Measures guest binding affinity by monitoring displacement from a host-indicator complex
- **IDA (Indicator Displacement Assay)**: Determines binding constants using indicator dye displacement
- **DBA (Direct Binding Assay)**: Direct measurement of host-guest or dye-host binding interactions
- **Dye Alone**: Linear calibration fitting for indicator dyes

The application features an intuitive Tkinter-based graphical user interface with data visualization, statistical analysis, and result export capabilities. 

## 🚀 Features

- **Multiple Assay Types**: Support for GDA, IDA, DBA (both host-to-dye and dye-to-host titrations), and dye alone calibration
- **Advanced Fitting Algorithms**: Non-linear least squares fitting with robust error estimation
- **Data Visualization**: Plots with unified styling and export options
- 🚧 **Statistical Analysis**: Comprehensive error analysis and confidence intervals
- **Result Merging**: Combine multiple fitting results for comprehensive analysis
- **Cross-Platform**: Available for Windows, macOS, and Linux
- 🚧 **Jupyter Integration**: Included notebooks for advanced analysis and method development

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Operating System: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- RAM: 2GB minimum, 4GB recommended
- Storage: 500MB free space

### Python Dependencies
```
matplotlib>=3.5.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
```

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
3. **Load your data** using the file browser (supports .txt files, see `./example_data`)
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

#### 🚧 Jupyter Notebooks
Explore the `notebooks/` directory for:
- Method development examples
- Advanced data analysis workflows
- Custom fitting procedures

## 📁 Project Structure

```
fitting_app/
├── main.py                   # Application entry point
├── requirements.txt
├── README.md                 # This file
├── AppIcon.png
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

The application implements several binding models based on mass action kinetics:

#### Direct Binding Assay (DBA)

DBA measures direct binding interactions through spectroscopic signal changes.

Binding Model:
```
H + D ⇌ HD
Ka(d) = [HD] / ([H][D])
```
#### Competitive Binding (IDA & GDA)

IDA determines binding constants using competitive displacement of an indicator dye by an analyte. GDA determines binding constants using competitive displacement of an analyte by an indicator dye.

Binding model:
```
H + D ⇌ HD    (Ka(d))
H + G ⇌ HG    (Ka(g))
```

Where displacement occurs according to relative binding affinities.

### Statistical Analysis

- **Non-linear least squares fitting** using Brent method
- **Residual analysis** for model validation
- 🚧 **Bootstrap error estimation** for parameter uncertainties

## 🔧 Building from Source

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

## 🧪 Testing

🚧 Unittest suite coming soon 

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
- Add docstrings to all functions and classes following [Numpy's style guide]()
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

### Common Issues

**Fitting convergence issues:**
- Check data quality and remove outliers
- Adjust initial parameter estimates
- Verify data format matches expected structure

### Getting Help

- **Issues**: Report bugs via [GitHub Issues](../../issues)
- **Discussions**: Join conversations in [GitHub Discussions](../../discussions)
- **Documentation**: 🚧 Coming soon

## 📄 License

This project is licensed under the GPL 3.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

🚧 WIP

---

**Version**: 0.4

**Last Updated**: June 2025

**Contact:** contact@suprabank.org