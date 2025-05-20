# Pattern Electroretinogram Classification using Gaussian Naive Bayes

This project uses the [PERG IOBA dataset](https://www.nature.com/articles/s41597-024-03125-y) to classify patients as having **normal** or **abnormal** eye conditions using features extracted from electroretinogram signals. The dataset consists of comprehensive ocular electrophysiology records. This classification is achieved using a Gaussian Naive Bayes model implemented in Python.

---

## 📁 Project Structure

Sample2/
├── index.py # Main analysis and model script
├── README.md # Project documentation
├── confusion_matrix.png # Generated confusion matrix plot
├── sample2/
│ └── a-comprehensive-dataset-of-pattern-electroretinograms...
│ └── csv/
│ ├── participants_info.csv
│ ├── 0001.csv
│ ├── 0002.csv
│ └── ...

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone git@github.com:Titophil/Sample2.git
cd Sample2
2. Install dependencies
bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn
3. Run the script
bash
Copy
Edit
python index.py
🧪 Methodology
Feature Extraction:

For each patient, features were extracted from the RE_1 and LE_1 columns in the signal CSV files.

Extracted features:

Mean

Standard Deviation

Minimum

Maximum

Skewness

Kurtosis

Labeling:

Patients with diagnosis1 == "Normal" were labeled 0.

All others were labeled 1 (abnormal).

Model:

Gaussian Naive Bayes classifier.

Data was split using an 80/20 ratio for training and testing.

Evaluation:

Accuracy

Precision, Recall, F1-score

Confusion Matrix Heatmap (saved as confusion_matrix.png)

📊 Results
Classification Report
Class	Precision	Recall	F1-Score	Support
Normal (0)	0.54	0.52	0.53	25
Abnormal (1)	0.73	0.74	0.74	43
Accuracy			0.66	68


🧠 Potential Improvements
Apply feature scaling or normalization.

Try other classifiers (Random Forest, SVM, XGBoost).

Use cross-validation for more robust evaluation.

Add domain-specific features (e.g., temporal signal patterns).

📚 Dataset Citation
Rodríguez, E., Lillo, J., Muñoz-García, A. et al. A comprehensive dataset of pattern electroretinograms for ocular electrophysiology research: the PERG IOBA dataset. Sci Data 11, 253 (2024). https://doi.org/10.1038/s41597-024-03125-y

👨‍💻 Author
Titus Kiprono
Passionate about machine learning in medical applications.
For questions or feedback, feel free to reach out!
kiprontitus254@gmail.com
