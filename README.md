# GPT-2 Finetuned Model

This model follows the standard GPT-2 Architecture

It uses:

- GPT-2 BPE tokenizer 

- GPT-2 Model

- GPT-2 Pretrained Weights

- All training data was taken from Hugging Face and transformed using `transform.py` compiled using `collate.py`

- 1 Epoch of Trainig

> NOTE
> I did not filter the data in any way (hindsight is 20/20 so shaddup) when I probably should've.
> By filtering I also mean modifying and creating multiple ways to say the same thing which probably has a better effect on model learning.

```
                  Tokenized Text
                        |
                        v
    +----------------------------------------+
    |                   |                    |
    |       +-----------------------+        |
    |       | Token Embedding Layer |        |
    |       +-----------------------+        |
    |                   |                    |
    |                   v                    |
    |     +----------------------------+     |
    |     | Positional Embedding Layer |     |
    |     +----------------------------+     |
    |                   |                    |
    |                   v                    |
    |              +---------+               |
    |              | Dropout |               |
    |              +---------+               |
    |                   |                    |
    |   +--------------------------------+   |  }
    |   |               |-------------   |   |  }   Transformer Block x 12
    |   |               v             \  |   |  }
    |   |    +-----------------------+ | |   |  }
    |   |    | Layer Normalization 1 | | |   |  }
    |   |    +-----------------------+ | |   |  }
    |   |               |              | |   |  }
    |   |               v              | |   |  }
    |   |    +----------------------+  | |   |  }
    |   |    | Multi-Head Attention |  | |   |  }
    |   |    +----------------------+  | |   |  }
    |   |               |              | |   |  }
    |   |               v              | |   |  }
    |   |          +---------+         | |   |  }
    |   |          | Dropout |         | |   |  }
    |   |          +---------+        /  |   |  }
    |   |               |<------------   |   |  }
    |   |    -----------|                |   |  }
    |   |   /           v                |   |  }
    |   |  | +-----------------------+   |   |  }
    |   |  | | Layer Normalization 2 |   |   |  }
    |   |  | +-----------------------+   |   |  }
    |   |  |            |                |   |  }
    |   |  |            v                |   |  }
    |   |  |    +--------------+         |   |  }
    |   |  |    | Feed Forward |         |   |  }
    |   |  |    +--------------+         |   |  }
    |   |  |            |                |   |  }
    |   |  |            v                |   |  }
    |   |  |       +---------+           |   |  }
    |   |  |       | Dropout |           |   |  }
    |   |   \      +---------+           |   |  }
    |   |    ---------->|                |   |  }
    |   +--------------------------------+   |  }
    |                   |                    |
    |                   v                    |
    |      +---------------------------+     |
    |      | Final Layer Normalization |     |
    |      +---------------------------+     |
    |                   |                    |
    |                   v                    |
    |          +---------------------+       |
    |          | Linear Output Layer |       |
    |          +---------------------+       |
    |                   |                    |
    +----------------------------------------+
                        |
                        v
                  Output Tokens
```

## Deployment

1. Install Python 3.10

2. Create a virtual environment

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment

    - Windows
        ```Powershell
        ./venv/Scripts/activate
        ```
    - Mac and Linux
        ```bash
        source ./venv/bin/activate
        ```
4. Install required libraries

    ```
    pip install -r requirements.txt
    ```

5. Train model

```bash
python ./train_model.py
```

6. Use model

```bash
python ./main.py
```

## Undeploy

1. Type in `exit` for graceful exit or press Ctrl + C for forced exit

2. Type in `deactivate` in the terminal to exit the virtual environment

### Sample Responses

```
> {user_question}
{AI Answer}
```

```
> what is the output for Arachnoiditis?
Arachnoiditis is a chronic, or long lasting, disease of the central nervous system. 
It is characterized by the sudden, severe symptoms of arachnoiditis. 
The most common symptoms are a sudden, severe headache that does not go away and back pain 
that does not go away. Other symptoms include a sudden, severe headache that does not go away
and back pain that does not go away. Arachnoiditis can be caused by any of many conditions, 
including viral infections, autoimmune diseases, cancer, inflammatory diseases, and some 
medications. Treatment depends on the cause of the arachnoiditis.
```

```
> What is the outlook for Learning Disabilities?
Learning disabilities can be a lifelong condition. Some people with learning disabilities 
have a normal life expectancy. Others may experience temporary loss of function.
```

```
> What research is being done for Learning Disabilities?
The National Institute of Allergy and Infectious Diseases (NIAID) conducts research in its 
laboratories at the National Institutes of Health (NIH) and also supports additional research
through grants to major research institutions across the country. Current research is 
exploring how the immune system malfunctions and causes disabilities. Additional research 
is focused on finding better ways to prevent, treat, and ultimately cure disorders such as 
learning disabilities.
```

```
> What are the treatments for spasticity?  
These resources address the diagnosis or management of spasticity:  - Gene Review: Gene 
Review: Spasticity  - Genetic Testing Registry: Spasticity  - Spasticity Support 
International: Spasticity Support International   These resources from MedlinePlus offer 
information about the diagnosis and management of various health conditions:  - Diagnostic 
Tests  - Drug Therapy  - Surgery and Rehabilitation  - Genetic Counseling   - Palliative 
Care
```
