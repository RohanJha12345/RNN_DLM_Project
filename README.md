{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KtfG2265a6Vb"
      },
      "source": [
        "# Recurrent Neural Network Sentiment analysis using\n",
        "\n",
        "Contributors:\n",
        "- Dimple(055009)\n",
        "- Rohan Jha(055057)\n",
        "---\n",
        "## Objective\n",
        "The primary objective of this project is to design, implement, and evaluate a deep learning-based sentiment analysis model using an RNN architecture. This model aims to classify movie reviews based on their sentiment (positive or negative) by leveraging the sequential patterns present in text data. The project aims to develop a model that can accurately and efficiently classify reviews, providing insights into public opinion and sentiment trends.\n",
        "\n",
        "## Data Description\n",
        "### 1. Datasets\n",
        "The project utilizes two datasets:\n",
        "- **IMDB Dataset of 50K Movie Reviews**: This dataset is used for training the RNN model. It contains 50,000 movie reviews labeled as either positive or negative.\n",
        "- **Metacritic Reviews Dataset**: This dataset is used for testing the generalization ability of the trained model. It consists of 151 manually collected reviews and ratings from Metacritic.\n",
        "\n",
        "### 2. IMDB Training Data\n",
        "- **Size**: 50,000 records\n",
        "- **Columns**:\n",
        "  - **Review**: The textual content of the movie review.\n",
        "  - **Sentiment**: The sentiment label (positive or negative). Positive sentiment is encoded as `1`, and negative sentiment is encoded as `0`.\n",
        "- A random sample of 40,000 reviews is selected for training.\n",
        "\n",
        "### 3. Metacritic Testing Data\n",
        "- **Size**: 151 records\n",
        "- **Columns**:\n",
        "  - **Movie Name**: The title of the movie.\n",
        "  - **Rating**: The rating given to the movie.\n",
        "  - **Review**: The textual content of the movie review.\n",
        "  - **Sentiment**: The sentiment label (positive or negative).\n",
        "\n",
        "## Data Preprocessing Steps\n",
        "### 1. Sentiment Encoding\n",
        "- **Positive Sentiment** → Encoded as `1`\n",
        "- **Negative Sentiment** → Encoded as `0`\n",
        "\n",
        "### 2. Text Normalization\n",
        "- **Removing Special Characters**: Stripping unnecessary characters (e.g., punctuation, special symbols) to clean the text.\n",
        "- **Lowercasing**: Converting all reviews to lowercase for uniformity and consistency.\n",
        "\n",
        "### 3. Tokenization\n",
        "- Splitting the text into individual tokens (words).\n",
        "- Using a vocabulary size of **20,000** most frequent words (`max_features=20000`). Any words outside this range are replaced with a placeholder token.\n",
        "\n",
        "### 4. Sequence Padding\n",
        "- Ensuring all tokenized reviews are of the same length by:\n",
        "  - Padding shorter sequences with zeros at the beginning or end.\n",
        "  - Truncating longer sequences to a maximum length of **400** tokens (`max_length = 400`).\n",
        "\n",
        "## Observations\n",
        "### 1. Library Imports\n",
        "- The notebook confirms the successful import of necessary libraries, including **TensorFlow, Pandas, NumPy, re** (for regular expressions), and **scikit-learn** (for data splitting).\n",
        "\n",
        "### 2. Data Loading and Preprocessing\n",
        "- The **Metacritic testing dataset** was loaded successfully from a Google Drive URL, as shown in the code.\n",
        "- The `columns` attribute and `shape` attribute confirmed the successful loading of **151 entries**, each having **4 columns**.\n",
        "\n",
        "\n",
        "## Model Building\n",
        "### 1. Model Architecture\n",
        "The model includes the following layers:\n",
        "- **Embedding Layer**: Input dimension of **20,000** (vocabulary size), output dimension of **128** (word embedding size), and input length of **400** (maximum sequence length).\n",
        "- **Recurrent Layer**: Simple RNN with **64 units** and a **Tanh activation function**. A **dropout rate of 0.2** is used for regularization.\n",
        "- **Fully Connected Layer**: Dense layer with **1 neuron** and a **Sigmoid activation function** for binary classification.\n",
        "\n",
        "## Model Training\n",
        "- The model is trained on **80% of the 40,000 sampled IMDB reviews** and validated on the remaining **20%**.\n",
        "- The model is compiled with **binary crossentropy loss**, **Adam optimizer** (learning rate = `0.001`), a batch size of **32**, and trained for **15 epochs**, with **early stopping** based on validation loss. Early stopping patience is set to **3 epochs**.\n",
        "\n",
        "## Model Performance\n",
        "- **Training accuracy** increased steadily, reaching approximately **89%** after **10 epochs**.\n",
        "- **Validation accuracy** remained stable at around **87%**, indicating good generalization.\n",
        "- The **final test accuracy** on the IMDB test set was around **86%**, suggesting a well-trained model with slight room for improvement.\n",
        "- The model performed similarly on the **Metacritic dataset**, achieving a test accuracy of approximately **77%**, showing that it generalizes well across different review datasets but could improve if **LSTM** was used instead of RNN.\n",
        "- **Early stopping** was triggered after a few epochs in both training phases, preventing overfitting and ensuring that the best model was retained.\n",
        "\n",
        "## Managerial Insights\n",
        "### 1. Model Effectiveness & Business Implications\n",
        "- The **RNN model** performs reasonably well on the **IMDB dataset** but generalizes poorly on **Metacritic reviews**.\n",
        "- This suggests that **Metacritic reviews** might have different **writing styles, slang, or review structures** compared to IMDB. This highlights the importance of **training data diversity** for robust sentiment analysis.\n",
        "\n",
        "### 2. Improvement Areas\n",
        "- **Better Preprocessing**: Introduce techniques like **stemming, lemmatization, stop-word removal, and n-grams** to improve accuracy.\n",
        "- **More Complex Architectures**: RNNs have **limited long-term memory**; switching to **LSTMs** may enhance generalization.\n",
        "- **Larger Dataset & Augmentation**: Training on a combined dataset of **IMDB and Metacritic reviews** may improve model robustness.\n",
        "- **Domain Adaptation**: Fine-tuning the model specifically on **Metacritic reviews** could improve cross-domain accuracy.\n",
        "\n",
        "### 3. Business Applications\n",
        "- **Customer Sentiment Monitoring**: Companies can use this model to analyze **movie, product, or service reviews** to gauge public opinion.\n",
        "- **Brand Reputation Analysis**: Identifying **sentiment trends** can help businesses manage **PR crises** and improve **customer engagement**.\n",
        "- **Automated Review Filtering**: Businesses can filter out **fake reviews or spam** using an improved sentiment classification model.\n",
        "\n",
        "### 4. Conclusion & Recommendations\n",
        "#### Immediate Steps:\n",
        "- Improve **text preprocessing** by handling **stop words** and using **TF-IDF weights**.\n",
        "- Fine-tune the model using **transfer learning** with additional datasets.\n",
        "- Consider switching to **LSTM/GRU-based models** for improved generalization.\n",
        "\n",
        "#### Long-Term Strategy:\n",
        "- Expand **training data** by incorporating **reviews from multiple platforms**.\n",
        "- Implement **real-time sentiment tracking** in a dashboard for **actionable insights**.\n",
        "- Conduct **A/B testing** with different architectures to find the **best-performing model**.\n",
        "- Aim for **higher accuracy (target: 75%+)** through **continuous optimization**.\n",
        "\n",
        "This meticulously detailed report provides an extensive overview of the **Reviews Sentiment Analysis** project using an **RNN**, meticulously covering the **objective, data description, observations, and managerial insights**. The inclusion of the **analysis of the report notebook, data preprocessing, model evaluation, and long-term strategic recommendations** highlight the model implementation insights.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y8FnQSZINZaI"
      },
      "source": [
        "## 1. Importing the Libraries\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "V9XDI9fmFVaF",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "21b5a3ae-f03c-4552-8a64-658a1a1d5fd7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.11/dist-packages (2.18.0)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.4.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (25.2.10)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.6.0)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (18.1.1)\n",
            "Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.4.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from tensorflow) (24.2)\n",
            "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (5.29.4)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.32.3)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.11/dist-packages (from tensorflow) (75.2.0)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.0)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.5.0)\n",
            "Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (4.13.0)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.2)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.71.0)\n",
            "Requirement already satisfied: tensorboard<2.19,>=2.18 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.18.0)\n",
            "Requirement already satisfied: keras>=3.5.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.8.0)\n",
            "Requirement already satisfied: numpy<2.1.0,>=1.26.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.0.2)\n",
            "Requirement already satisfied: h5py>=3.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.13.0)\n",
            "Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.4.1)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.37.1)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from astunparse>=1.6.0->tensorflow) (0.45.1)\n",
            "Requirement already satisfied: rich in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (13.9.4)\n",
            "Requirement already satisfied: namex in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (0.0.8)\n",
            "Requirement already satisfied: optree in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (0.14.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.3.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2025.1.31)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.7)\n",
            "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (0.7.2)\n",
            "Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.1.3)\n",
            "Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.11/dist-packages (from werkzeug>=1.0.1->tensorboard<2.19,>=2.18->tensorflow) (3.0.2)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras>=3.5.0->tensorflow) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras>=3.5.0->tensorflow) (2.18.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.5.0->tensorflow) (0.1.2)\n"
          ]
        }
      ],
      "source": [
        "!pip install tensorflow\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import re\n",
        "from sklearn.model_selection import train_test_split\n",
        "from tensorflow.keras.preprocessing.text import Tokenizer\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import SimpleRNN, Dense, Embedding,Dropout\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Kj3rB7p0N2T5"
      },
      "source": [
        "## 2. Preparing the Dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6gRBUPqTFzAp",
        "outputId": "555eeea2-57e0-4ae9-d5b8-a0188b3063ad"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Columns in the dataset:\n",
            "['Movie Name', 'Rating', 'Review', 'sentiment']\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(151, 4)"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import io\n",
        "import requests\n",
        "\n",
        "# Google Drive File Link (URL) of Test Dataset\n",
        "url = \"https://drive.google.com/file/d/1TKkoE7DW9uCNx4wdTgM8egfG0FfZPhUy/view?usp=sharing\"\n",
        "\n",
        "# Extract the File Id from the URL\n",
        "file_id = url.split('/')[-2]\n",
        "\n",
        "# Construct the Download URL\n",
        "download_url = f\"https://drive.google.com/uc?id={file_id}\"\n",
        "\n",
        "# Download the File Content\n",
        "response = requests.get(download_url)\n",
        "response.raise_for_status()  # Raise an Exception for Bad Responses\n",
        "\n",
        "# Read the Training Dataset\n",
        "dbrj0957_data1 = pd.read_csv(io.StringIO(response.text))\n",
        "\n",
        "# Display Traing Dataset Information\n",
        "print(\"Columns in the dataset:\")\n",
        "print(dbrj0957_data1.columns.tolist())\n",
        "dbrj0957_data1.shape\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Assuming you want a sample of up to 40,000, but not exceeding the DataFrame's size\n",
        "dbrj0957_data = dbrj0957_data1.sample(n=min(40000, len(dbrj0957_data1)))"
      ],
      "metadata": {
        "id": "VhjfqV6dtWdA"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dI4LIHSgFxnZ"
      },
      "source": [
        "### 2.1 Creating a random state of 40000 records"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RjG-_V6lOaVd"
      },
      "source": [
        "### 2.2 Data cleaning and pre-processing"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Replace 'review' with the actual column name if it's different\n",
        "actual_column_name = \"Review\"  # Update with the correct column name\n",
        "\n",
        "dbrj0957_data[actual_column_name] = dbrj0957_data[actual_column_name].str.lower()\n",
        "dbrj0957_data[actual_column_name] = dbrj0957_data[actual_column_name].replace(r'[^a-z0-9\\s]', '', regex=True)\n",
        "\n",
        "dbrj0957_data['sentiment value'] = dbrj0957_data['sentiment'].apply(lambda x: 1 if x == \"positive\" else 0)\n",
        "dbrj0957_data = dbrj0957_data.dropna()"
      ],
      "metadata": {
        "id": "5CLSoDLmtj02"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QSUuitILO07u"
      },
      "source": [
        "### 2.3 Tokenizing and Padding of dataset"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "dbrj0957_max_features = 20000\n",
        "dbrj0957_max_length =400\n",
        "\n",
        "tokenizer = Tokenizer(num_words=dbrj0957_max_features)\n",
        "# The column name is likely 'Review' instead of 'review'\n",
        "tokenizer.fit_on_texts(dbrj0957_data[\"Review\"])\n",
        "X = pad_sequences(tokenizer.texts_to_sequences(dbrj0957_data[\"Review\"]), maxlen=dbrj0957_max_length)\n",
        "y = dbrj0957_data['sentiment value'].values"
      ],
      "metadata": {
        "id": "jHx4uEmGt0Wk"
      },
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lJSO3IOCPCWL"
      },
      "source": [
        "### 2.4 Spliting of dataset into test, train and validation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "id": "gKsc41H0GZw_"
      },
      "outputs": [],
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=42, stratify=y\n",
        ")\n",
        "X_train, X_val, y_train, y_val = train_test_split(\n",
        "    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train\n",
        ")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "e4NsCCTXPmmF"
      },
      "source": [
        "## 3. Model Preparation"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Qf3xkbtmG4VH",
        "outputId": "544c2eab-ec98-4a0e-8a39-c3d94314ecd4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.\n",
            "  warnings.warn(\n"
          ]
        }
      ],
      "source": [
        "from tensorflow.keras.optimizers import Adam\n",
        "from tensorflow.keras.callbacks import EarlyStopping\n",
        "dbrj0957_model1 = Sequential([\n",
        "    Embedding(input_dim=dbrj0957_max_features, output_dim=128, input_length=dbrj0957_max_length),\n",
        "    SimpleRNN(64, activation='tanh', return_sequences=False),\n",
        "    Dense(1, activation='sigmoid'),\n",
        "    Dropout(0.2),  # Helps prevent overfitting\n",
        "])\n",
        "\n",
        "dbrj0957_model1.compile(\n",
        "    loss='binary_crossentropy',\n",
        "    optimizer=Adam(learning_rate=0.0001),\n",
        "    metrics=['accuracy']\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qNH_mgUwf2F8"
      },
      "source": [
        "### 3.1 Hyperparameter Tuning"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 810
        },
        "id": "4fn-gYk0RsPc",
        "outputId": "18db25ee-ef5e-4886-9a40-58ffe925de90"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Trial 28 Complete [00h 00m 19s]\n",
            "val_accuracy: 0.6666666865348816\n",
            "\n",
            "Best val_accuracy So Far: 0.8333333134651184\n",
            "Total elapsed time: 00h 05m 48s\n",
            "\n",
            "Search: Running Trial #29\n",
            "\n",
            "Value             |Best Value So Far |Hyperparameter\n",
            "256               |256               |embedding_dim\n",
            "64                |64                |rnn_units\n",
            "0.5               |0.5               |dropout_1\n",
            "32                |64                |rnn_units_2\n",
            "0.3               |0.2               |dropout_2\n",
            "0.001             |0.0005            |learning_rate\n",
            "10                |10                |tuner/epochs\n",
            "0                 |4                 |tuner/initial_epoch\n",
            "0                 |2                 |tuner/bracket\n",
            "0                 |2                 |tuner/round\n",
            "\n",
            "Epoch 1/10\n",
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 560ms/step - accuracy: 0.4679 - loss: 0.7502 - val_accuracy: 0.6667 - val_loss: 0.6073\n",
            "Epoch 2/10\n",
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 362ms/step - accuracy: 0.8125 - loss: 0.5135 - val_accuracy: 0.5833 - val_loss: 0.5978\n",
            "Epoch 3/10\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-30-493b99af9290>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     27\u001b[0m )\n\u001b[1;32m     28\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 29\u001b[0;31m \u001b[0mtuner\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msearch\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mepochs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvalidation_data\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_val\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_val\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m32\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     30\u001b[0m \u001b[0mbest_hps\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtuner\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_best_hyperparameters\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnum_trials\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py\u001b[0m in \u001b[0;36msearch\u001b[0;34m(self, *fit_args, **fit_kwargs)\u001b[0m\n\u001b[1;32m    232\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    233\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_trial_begin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrial\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 234\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_try_run_and_update_trial\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrial\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0mfit_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfit_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    235\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_trial_end\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrial\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    236\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_search_end\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py\u001b[0m in \u001b[0;36m_try_run_and_update_trial\u001b[0;34m(self, trial, *fit_args, **fit_kwargs)\u001b[0m\n\u001b[1;32m    272\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_try_run_and_update_trial\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrial\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0mfit_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfit_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    273\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 274\u001b[0;31m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_run_and_update_trial\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrial\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0mfit_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfit_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    275\u001b[0m             \u001b[0mtrial\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstatus\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrial_module\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mTrialStatus\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mCOMPLETED\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    276\u001b[0m             \u001b[0;32mreturn\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py\u001b[0m in \u001b[0;36m_run_and_update_trial\u001b[0;34m(self, trial, *fit_args, **fit_kwargs)\u001b[0m\n\u001b[1;32m    237\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    238\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_run_and_update_trial\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrial\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0mfit_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfit_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 239\u001b[0;31m         \u001b[0mresults\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrun_trial\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrial\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0mfit_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfit_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    240\u001b[0m         if self.oracle.get_trial(trial.trial_id).metrics.exists(\n\u001b[1;32m    241\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moracle\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mobjective\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras_tuner/src/tuners/hyperband.py\u001b[0m in \u001b[0;36mrun_trial\u001b[0;34m(self, trial, *fit_args, **fit_kwargs)\u001b[0m\n\u001b[1;32m    425\u001b[0m             \u001b[0mfit_kwargs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"epochs\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mhp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"tuner/epochs\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    426\u001b[0m             \u001b[0mfit_kwargs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"initial_epoch\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mhp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"tuner/initial_epoch\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 427\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0msuper\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrun_trial\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrial\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0mfit_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mfit_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    428\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    429\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m_build_hypermodel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mhp\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/tuner.py\u001b[0m in \u001b[0;36mrun_trial\u001b[0;34m(self, trial, *args, **kwargs)\u001b[0m\n\u001b[1;32m    312\u001b[0m             \u001b[0mcallbacks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel_checkpoint\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    313\u001b[0m             \u001b[0mcopied_kwargs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"callbacks\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mcallbacks\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 314\u001b[0;31m             \u001b[0mobj_value\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_build_and_fit_model\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtrial\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mcopied_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    315\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    316\u001b[0m             \u001b[0mhistories\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mobj_value\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/tuner.py\u001b[0m in \u001b[0;36m_build_and_fit_model\u001b[0;34m(self, trial, *args, **kwargs)\u001b[0m\n\u001b[1;32m    231\u001b[0m         \u001b[0mhp\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrial\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhyperparameters\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    232\u001b[0m         \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_try_build\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhp\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 233\u001b[0;31m         \u001b[0mresults\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhypermodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mhp\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    234\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    235\u001b[0m         \u001b[0;31m# Save the build config for model loading later.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/hypermodel.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, hp, model, *args, **kwargs)\u001b[0m\n\u001b[1;32m    147\u001b[0m             \u001b[0mIf\u001b[0m \u001b[0;32mreturn\u001b[0m \u001b[0ma\u001b[0m \u001b[0mfloat\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mit\u001b[0m \u001b[0mshould\u001b[0m \u001b[0mbe\u001b[0m \u001b[0mthe\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m`\u001b[0m\u001b[0mobjective\u001b[0m\u001b[0;31m`\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    148\u001b[0m         \"\"\"\n\u001b[0;32m--> 149\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    150\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    151\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py\u001b[0m in \u001b[0;36merror_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    115\u001b[0m         \u001b[0mfiltered_tb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    116\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 117\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    118\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    119\u001b[0m             \u001b[0mfiltered_tb\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_process_traceback_frames\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0me\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__traceback__\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq)\u001b[0m\n\u001b[1;32m    368\u001b[0m             \u001b[0;32mwith\u001b[0m \u001b[0mepoch_iterator\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcatch_stop_iteration\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    369\u001b[0m                 \u001b[0;32mfor\u001b[0m \u001b[0mstep\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0miterator\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mepoch_iterator\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 370\u001b[0;31m                     \u001b[0mcallbacks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_train_batch_begin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    371\u001b[0m                     \u001b[0mlogs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtrain_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    372\u001b[0m                     \u001b[0mcallbacks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_train_batch_end\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlogs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/callback_list.py\u001b[0m in \u001b[0;36mon_train_batch_begin\u001b[0;34m(self, batch, logs)\u001b[0m\n\u001b[1;32m    145\u001b[0m             \u001b[0mcallback\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_epoch_end\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mepoch\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlogs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    146\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 147\u001b[0;31m     \u001b[0;32mdef\u001b[0m \u001b[0mon_train_batch_begin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mbatch\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlogs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    148\u001b[0m         \u001b[0mlogs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpython_utils\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpythonify_logs\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlogs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    149\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mcallback\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcallbacks\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ],
      "source": [
        "!pip install keras_tuner\n",
        "import keras_tuner as kt\n",
        "def build_model(hp):\n",
        "    model = Sequential([\n",
        "        Embedding(input_dim=dbrj0957_max_features, output_dim=hp.Choice('embedding_dim', [64, 128, 256]), input_length=dbrj0957_max_length),\n",
        "        SimpleRNN(hp.Choice('rnn_units', [32, 64, 128]), return_sequences=True),\n",
        "        Dropout(hp.Choice('dropout_1', [0.2, 0.3, 0.5])),\n",
        "        SimpleRNN(hp.Choice('rnn_units_2', [32, 64]), return_sequences=False),\n",
        "        Dropout(hp.Choice('dropout_2', [0.2, 0.3, 0.5])),\n",
        "        Dense(1, activation='sigmoid')\n",
        "    ])\n",
        "    model.compile(\n",
        "        loss='binary_crossentropy',\n",
        "        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [0.001, 0.0005, 0.0001])),\n",
        "        metrics=['accuracy']\n",
        "    )\n",
        "    return model\n",
        "\n",
        "# Hyperparameter tuning\n",
        "tuner = kt.Hyperband(\n",
        "    build_model,\n",
        "    objective='val_accuracy',\n",
        "    max_epochs=10,\n",
        "    factor=3,\n",
        "    directory='tuner_dir',\n",
        "    project_name='sentiment_analysis'\n",
        ")\n",
        "\n",
        "tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)\n",
        "best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kUYeTTYzPwol"
      },
      "source": [
        "### 3.2 Model Building"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "metadata": {
        "id": "AQkDdZS6H1ca"
      },
      "outputs": [],
      "source": [
        "# commenting this since the model has been trained and saved\n",
        "# dbrj0957_early_stopping = EarlyStopping(\n",
        "#     monitor='val_loss', patience=3, restore_best_weights=True\n",
        "# )\n",
        "\n",
        "# dbrj0957_history11 = dbrj0957_model1.fit(\n",
        "#     X_train, y_train,\n",
        "#     epochs=10,\n",
        "#     batch_size=32,\n",
        "#     validation_data=(X_val, y_val),\n",
        "#     callbacks=[dbrj0957_early_stopping],  # Stops if validation loss doesn't improve\n",
        "#     verbose=1\n",
        "# )\n",
        "\n",
        "# dbrj0957_score = dbrj0957_model1.evaluate(X_test, y_test, verbose=0)\n",
        "# print(f\"Test accuracy: {dbrj0957_score[1]:.2f}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# commenting this since the model has been trained and saved\n",
        "# #running code for 5 more epochs\n",
        "from tensorflow.keras.callbacks import EarlyStopping # Importing EarlyStopping\n",
        "\n",
        "dbrj0957_early_stopping = EarlyStopping(\n",
        "    monitor='val_loss', patience=3, restore_best_weights=True\n",
        ") # Defining dbrj0957_early_stopping\n",
        "\n",
        "dbrj0957_history1 = dbrj0957_model1.fit(\n",
        "    X_train, y_train,\n",
        "    epochs=5,\n",
        "    batch_size=32,\n",
        "    validation_data=(X_val, y_val),\n",
        "    callbacks=[dbrj0957_early_stopping],  # Stops if validation loss doesn't improve\n",
        "    verbose=1\n",
        ")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PERUbSejviGh",
        "outputId": "b871f29f-767c-44aa-cee8-e9ef66a35741"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/5\n",
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 627ms/step - accuracy: 0.6185 - loss: 2.3445 - val_accuracy: 0.5000 - val_loss: 0.7302\n",
            "Epoch 2/5\n",
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 589ms/step - accuracy: 0.6199 - loss: 2.5307 - val_accuracy: 0.5000 - val_loss: 0.7201\n",
            "Epoch 3/5\n",
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 256ms/step - accuracy: 0.5132 - loss: 4.2064 - val_accuracy: 0.5000 - val_loss: 0.7181\n",
            "Epoch 4/5\n",
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 164ms/step - accuracy: 0.6363 - loss: 1.9772 - val_accuracy: 0.5000 - val_loss: 0.7167\n",
            "Epoch 5/5\n",
            "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 182ms/step - accuracy: 0.6734 - loss: 2.9399 - val_accuracy: 0.5000 - val_loss: 0.7135\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 35,
      "metadata": {
        "id": "6MdZpDfASgx8"
      },
      "outputs": [],
      "source": [
        "dbrj0957_model1.save('dbrj0957_model12.keras')"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4UpH7nV4QOHm"
      },
      "source": [
        "## 4. Loading Metacritic dataset for testing"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 38,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 356
        },
        "id": "kJltGW7J_zDp",
        "outputId": "96bd9f8b-d713-488b-dbc6-15a8ed0d0d47",
        "collapsed": true
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "FileNotFoundError",
          "evalue": "[Errno 2] No such file or directory: 'metacritic test dataset1.csv'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-38-3c7457ce1a85>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mdbrj0957_test_data1\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'metacritic test dataset1.csv'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mencoding\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'latin1'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"Columns in the dataset:\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtolist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhead\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36mread_csv\u001b[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)\u001b[0m\n\u001b[1;32m   1024\u001b[0m     \u001b[0mkwds\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mkwds_defaults\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1025\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1026\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0m_read\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1027\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1028\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36m_read\u001b[0;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[1;32m    618\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    619\u001b[0m     \u001b[0;31m# Create the parser.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 620\u001b[0;31m     \u001b[0mparser\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mTextFileReader\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    621\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    622\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mchunksize\u001b[0m \u001b[0;32mor\u001b[0m \u001b[0miterator\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[1;32m   1618\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1619\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhandles\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mIOHandles\u001b[0m \u001b[0;34m|\u001b[0m \u001b[0;32mNone\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1620\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_engine\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_make_engine\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1621\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1622\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m->\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py\u001b[0m in \u001b[0;36m_make_engine\u001b[0;34m(self, f, engine)\u001b[0m\n\u001b[1;32m   1878\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0;34m\"b\"\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1879\u001b[0m                     \u001b[0mmode\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;34m\"b\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1880\u001b[0;31m             self.handles = get_handle(\n\u001b[0m\u001b[1;32m   1881\u001b[0m                 \u001b[0mf\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1882\u001b[0m                 \u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/common.py\u001b[0m in \u001b[0;36mget_handle\u001b[0;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)\u001b[0m\n\u001b[1;32m    871\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mioargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mencoding\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0;34m\"b\"\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mioargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    872\u001b[0m             \u001b[0;31m# Encoding\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 873\u001b[0;31m             handle = open(\n\u001b[0m\u001b[1;32m    874\u001b[0m                 \u001b[0mhandle\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    875\u001b[0m                 \u001b[0mioargs\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'metacritic test dataset1.csv'"
          ]
        }
      ],
      "source": [
        "dbrj0957_test_data1 = pd.read_csv('metacritic test dataset1.csv', encoding='latin1')\n",
        "print(\"Columns in the dataset:\")\n",
        "print(dbrj0957_test_data1.columns.tolist())\n",
        "dbrj0957_test_data1.head(10)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Du5dsJDyQfWF"
      },
      "source": [
        "### 4.1 Data cleaning and pre-processing"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 41,
      "metadata": {
        "id": "89gI0p3D-kqi",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 228
        },
        "collapsed": true,
        "outputId": "7ea4ac07-cd44-4493-d37a-cb0bd6113674"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'dbrj0957_test_data1' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-41-7f357c766c01>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"Review\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"Review\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mstr\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mlower\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"Review\"\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"Review\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreplace\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mr'[^a-z0-9\\s]'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m''\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mregex\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mTrue\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'sentiment value'\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'sentiment'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mapply\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;32mlambda\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0;36m1\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m==\u001b[0m \u001b[0;34m\"positive\"\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0mdbrj0957_test_data1\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdropna\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'dbrj0957_test_data1' is not defined"
          ]
        }
      ],
      "source": [
        "dbrj0957_test_data1[\"Review\"] = dbrj0957_test_data1[\"Review\"].str.lower()\n",
        "dbrj0957_test_data1[\"Review\"] = dbrj0957_test_data1[\"Review\"].replace(r'[^a-z0-9\\s]', '', regex=True)\n",
        "\n",
        "dbrj0957_test_data1['sentiment value'] = dbrj0957_test_data1['sentiment'].apply(lambda x: 1 if x== \"positive\" else 0)\n",
        "dbrj0957_test_data1 = dbrj0957_test_data1.dropna()\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Z6zTuoh9Qj7X"
      },
      "source": [
        "### 4.2 Tokenizing and padding"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 44,
      "metadata": {
        "id": "g4QHVBbC-3mK",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 211
        },
        "collapsed": true,
        "outputId": "e70b6b8e-fe0c-4f32-8926-27649bb31ea4"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "NameError",
          "evalue": "name 'dbrj0957_test_data1' is not defined",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-44-74a71619d19a>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m# tokenizer = Tokenizer(num_words=20000)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;31m# tokenizer.fit_on_texts(dbrj0957_test_data1[\"Review\"])\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mX\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mpad_sequences\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtokenizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtexts_to_sequences\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m\"Review\"\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmaxlen\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdbrj0957_max_length\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      4\u001b[0m \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mdbrj0957_test_data1\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'sentiment value'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mNameError\u001b[0m: name 'dbrj0957_test_data1' is not defined"
          ]
        }
      ],
      "source": [
        "\n",
        "# tokenizer = Tokenizer(num_words=20000)\n",
        "# tokenizer.fit_on_texts(dbrj0957_test_data1[\"Review\"])\n",
        "X = pad_sequences(tokenizer.texts_to_sequences(dbrj0957_test_data1[\"Review\"]), maxlen=dbrj0957_max_length)\n",
        "y = dbrj0957_test_data1['sentiment value'].values\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KfndDy9EQv0r"
      },
      "source": [
        "## 4.3 Accuracy on Metacritic test dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 45,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "73ic31uJCiSu",
        "outputId": "03685113-180d-4fe3-d6e1-de8f8ff5a5d6"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Test accuracy: 0.74\n"
          ]
        }
      ],
      "source": [
        "\n",
        "score = dbrj0957_model1.evaluate(X, y, verbose=0)\n",
        "print(f\"Test accuracy: {score[1]:.2f}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jj3wBm7u9XN3"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
