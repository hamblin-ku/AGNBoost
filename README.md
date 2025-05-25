<div align="center">
  <table>
    <tr>
      <td align="left">
        <h1>AGNBoost: A machine learning toolkit for astronomical data analysis using XGBoost</h1>
      </td>
      <td align="center" width="220">
        <img src="figures/agnboost_logo.png" alt="AGNBoost Logo" width="200" height="200"/>
      </td>
    </tr>
  </table>
  
  [![Documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://hamblin-ku.github.io/AGNBoost/)
  [![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

AGNBoost is a machine learnign framework focused on efficiently identfying populations of candidates Active Galactic Nuclei (AGN) from photometric data. AGNBoost is built on [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/), an extension of the well known XGBoost algorithm to distributional modeling. The framework comes with pre-trained models for AGN identification and photometric redshift estimation from JWST NIRCam+MIRI photometric data, trained on catalogs of mock galaxies from [CIGALE](https://cigale.lam.fr/).


## ✨ Features
<ul>
  <li> Complete distirbutional modeling with XGBoostlss </li> 
  <li> Robust estimates of both model uncertainty and uncertainty and uncertainty due to photometric error</li> 
  <li> Flexible framework that allows: </li> 
  <ul>
    <li> Simple model creation and re-training </li> 
    <li> Straightforward incorporation of additional photometric bands and target variables </li> 
  </ul>
  <li> Missing photometric data imputation with SGAIN </li> 
</ul>

## 🚀 Installation

### Prerequisites
- Python 3.10 or later
- Git

### Quick Install

```bash
# Clone the repository
git clone https://github.com/kurthamblin/agnboost.git
cd agnboost

# Install in development mode (recommended)
pip install -e .
```

### Using Virtual Conda Environment (Recommended)

```bash
# Create and activate a conda environment
conda create -n agnboost python=3.11
conda activate agnboost

# Clone and install
git clone https://github.com/kurthamblin/agnboost.git
cd agnboost
pip install -e .
```


## 📖 Documentation

Complete documentation is available on GitHub Pages:

**🌐 [https://kurthamblin.github.io/agnboost/](https://kurthamblin.github.io/agnboost/)**

The documentation includes:
- **Examples**: Example jupyter notebooks illustrating how to use AGNBoost modules
- **API Reference**: Complete documentation of all classes and methods

## 🤝 Contributing

We welcome contributions from the astronomy community! Please see our [Contributing Guide](https://kurthamblin.github.io/agnboost/contributing/) for details on how to get involved.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [XGBoostLSS](https://statmixedml.github.io/XGBoostLSS/) for distributional modeling
- Uses [SGAIN](https://github.com/dtneves/ICCS_2021) for photometric imputation
- [CIGALE](https://cigale.lam.fr/) used for mock galaxy catalog creation
- Developed by astronomers, for astronomers

---

<div align="center">
  <strong>If you use AGNBoost in your research, please see our <a href="https://kurthamblin.github.io/agnboost/citation/">citation guide</a>.</strong>
</div>
```

