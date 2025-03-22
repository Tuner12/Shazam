
<p align="center">
  <img src="logo_Shazam.jpg" alt="Shazam Logo" width="120"/>
</p>

<h2 align="center">Shazam</h2>

<p align="center">
  A lightweight model for feature knowledge distillation using histopathology foundational models.
</p>

---

## 📌 Project Overview

**Shazam** proposes a small and efficient model that distills knowledge from extracted features using histopathology foundational models. This approach effectively leverages the strong representational power of large-scale foundational models while optimizing computational efficiency through a lightweight distillation process.

### ✅ Key Highlights:

- **Feature Knowledge Distillation**  
  Transfers rich representations from foundational models into a smaller, more efficient model.

- **Lightweight and Scalable**  
  Achieves high accuracy with lower computational cost, suitable for practical deployment in clinical settings.

- **Superior Performance**  
  Outperforms existing CPath models and other fusion-based methods across multiple evaluation benchmarks.

---

## 📂 Project Structure

![Project Structure](framework.png)

1. **Feature Extraction**: Leverages pretrained foundational histopathology models to extract high-level features from images.  
2. **Knowledge Distillation**: A small model learns to replicate the representational power of the foundational models.  
3. **Model Evaluation**: The distilled model is evaluated and compared against existing methods like CPath.

---




## ⚙️ Environment Setup

We directly use the environment configuration provided by the [**CLAM** project](https://github.com/mahmoodlab/CLAM).

### 1. Create the Conda Environment
```bash
conda env create -f env.yml
```

### 2. Activate the Environment
```bash
conda activate clam_latest
```





## 3. Train the Model
```bash
python train.py 
```




## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for more details.

---

If you find this project helpful, please consider giving it a ⭐️ on GitHub!
```
