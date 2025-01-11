# Spam Classifier 

## Overview

This project implements a Bayesian Spam Classifier to identify spam messages based on the content of SMS messages. Using probabilities derived from the dataset, the model predicts whether a message is spam or ham (not spam). The implementation includes preprocessing, training, testing, and optimization steps.

---

## Dataset
- **Source**: SMS Spam Collection Dataset from UCI.
- **Structure**:
  - `label`: Indicates whether a message is "spam" or "ham".
  - `message`: The actual content of the SMS.
  
---

## Dependencies
The following Python libraries are required:
- pandas
- numpy
- nltk
- scikit-learn

Install the dependencies using:
```bash
pip install pandas numpy nltk scikit-learn
```

---

## Implementation Steps

### 1. Import Libraries
The project utilizes libraries such as `pandas` for data manipulation, `nltk` for text processing, and `scikit-learn` for model evaluation.

### 2. Load and Prepare the Dataset
The dataset is loaded from a CSV file, and a binary label is created to distinguish between spam (`1`) and ham (`0`).

### 3. Preprocess the Messages
- Tokenize messages into words using NLTK.
- Convert text to lowercase and remove stop words and non-alphanumeric characters.

### 4. Split Dataset
The dataset is split into training (80%) and testing (20%) sets using scikit-learn's `train_test_split` function.

### 5. Calculate Probabilities
- **Prior Probabilities**:
  - \( P(\text{spam}) \): Proportion of spam messages.
  - \( P(\text{ham}) \): Proportion of ham messages.
- **Word Likelihoods**:
  - \( P(\text{word}|\text{spam}) \): Frequency of each word in spam messages.
  - \( P(\text{word}|\text{ham}) \): Frequency of each word in ham messages.

### 6. Classify Messages
Using Bayes' Theorem:
\[
P(\text{spam}|\text{words}) = \frac{P(\text{words}|\text{spam}) \cdot P(\text{spam})}{P(\text{words})}
\]
Messages are classified as spam if the posterior probability \( P(\text{spam}|\text{words}) \) is higher than \( P(\text{ham}|\text{words}) \).

### 7. Evaluate the Model
Accuracy is calculated using:
\[
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]
The current model achieves **98.83% accuracy**.

---

## How to Run the Project
1. **Download NLTK Data**:
   Run the following code to download necessary NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

2. **Run the Script**:
   Execute the script containing the implementation to train and test the model.

3. **Modify the Dataset**:
   Replace the dataset in `./data/data.csv` with your own SMS data for custom classification.

---

## Results
- **Accuracy**: ~98.83%
- **Error Cases**:
  - Messages with rare words or ambiguous intent are occasionally misclassified.

---

## Optimization Suggestions
To improve the accuracy:
1. **Feature Engineering**:
   - Add n-grams (e.g., bigrams or trigrams).
   - Incorporate additional features such as message length or the presence of links.

2. **Preprocessing**:
   - Use stemming or lemmatization to reduce word redundancy.

3. **Switch Models**:
   - Experiment with machine learning algorithms like SVM, Random Forest, or Logistic Regression.
   - Use vectorization techniques like TF-IDF.

4. **Data Augmentation**:
   - Enrich the dataset with more labeled examples to improve generalization.

---

## Example Output
```
Data has been successfully converted to ./data/data.csv
P(spam): 0.13, P(ham): 0.87
P('free'|spam): 0.0001
P('free'|ham): 0.0047
Accuracy: 98.83%
```

---

## Contact
For questions or improvements, feel free to reach out!

