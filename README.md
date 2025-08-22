# Rule-Based Model Generation and Interpretability

This repository demonstrates a complete machine learning pipeline that
integrates deep learning with interpretable rule-based models.\
It combines a **Neural Network** for high-performance classification
with **RuleKit** for generating human-understandable rules.

------------------------------------------------------------------------

## **Project Overview**

This project focuses on combining **predictive modeling** with **model
interpretability**:

-   Preprocesses datasets and prepares features for training.
-   Trains a **Neural Network** using TensorFlow/Keras for
    classification.
-   Uses a custom **RulekitWrapper** to extract interpretable rules from
    predictions.
-   Evaluates both neural network performance and rule-based model
    fidelity.
-   Provides **local explainability** for individual samples.

------------------------------------------------------------------------

## **Key Features**

### **1. Neural Network Classifier**

-   Multi-layer perceptron architecture.
-   Uses **early stopping** to avoid overfitting.
-   Produces high-accuracy predictions.

### **2. RuleKit Integration**

-   Uses a custom `RulekitWrapper` class.
-   Extracts **human-readable rules** from neural network outputs.
-   Generates rule statistics like coverage, support, and confidence.
-   Provides **fidelity scores** to measure alignment between rules and
    the neural model.

### **3. Local Explainability**

-   Explains **which rules** influence an individual prediction.
-   Highlights interpretable decision-making paths.

------------------------------------------------------------------------

## **Installation**

### **1. Clone the Repository**

``` bash
git clone https://github.com/AbdullahSajid35/RulekitWrapper
cd RulekitWrapper
```

### **2. Create a Conda Environment**

``` bash
conda create -n rulekit_env
conda activate rulekit_env
```

### **3. Install Dependencies**

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## **Usage**

### **Run the Main Script**

``` bash
python main.py
```

### **Pipeline Steps**

-   Loads and preprocesses data.
-   Trains a neural network classifier.
-   Extracts interpretable rules using `RulekitWrapper`.
-   Displays:
    -   Neural network performance.
    -   Rule-based statistics.
    -   Fidelity score.
    -   Local explanations.

------------------------------------------------------------------------

## **RuleKitWrapper Overview**

The `RulekitWrapper` class provides a unified interface for training,
extracting, and explaining rules.

### **Initialization**

``` python
RulekitWrapper(max_growing=2, max_rule_count=3)
```

------------------------------------------------------------------------

## **Expected Outputs**

-   **Neural Network Performance**
    -   Training accuracy and balanced accuracy.
    -   Test accuracy and balanced accuracy.
-   **Rule-Based Model Insights**
    -   Number of rules generated.
    -   Rule weights and importance.
    -   Rule coverage and support.
    -   Fidelity score.
-   **Local Explainability**
    -   Human-readable explanations for individual samples.

------------------------------------------------------------------------

## **Folder Structure**

    project/
    │── Datasets/              # Place your dataset(s) here
    │── RulekitWrapper.py      # Custom wrapper for RuleKit
    │── main.py                # Main pipeline script
    │── requirements.txt       # Project dependencies
    │── README.md              # Project documentation

------------------------------------------------------------------------

## **Dependencies**

-   Python ≥ 3.8
-   NumPy
-   Pandas
-   Scikit-learn
-   TensorFlow / Keras
-   RuleKit
-   Matplotlib (optional, for visualizations)

------------------------------------------------------------------------

## **License**

This project is open-source and available under the MIT License.

------------------------------------------------------------------------

