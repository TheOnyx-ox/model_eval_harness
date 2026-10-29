## Results

| Model | Rows | Accuracy (%) | 95% margin | Macro-F1 | Invalid | Unprocessed |
| --- | --- | --- | --- | --- | --- | --- |
| gemini | 500 | 60.0 | ±4.3 | 0.511 | 0 | 0 |
| groq | 500 | 55.6 | ±4.4 | 0.482 | 0 | 0 |
| openai | 500 | 57.6 | ±4.3 | 0.486 | 0 | 0 |

### Recall per emotion

| Model | sadness | joy | love | anger | fear | surprise |
| --- | --- | --- | --- | --- | --- | --- |
| gemini | 0.68 | 0.68 | 0.26 | 0.56 | 0.56 | 0.31 |
| groq | 0.63 | 0.54 | 0.39 | 0.57 | 0.54 | 0.39 |
| openai | 0.69 | 0.62 | 0.33 | 0.49 | 0.53 | 0.23 |
