import csv

from model_eval_harness.evaluator import label_to_id, build_summary, score_predictions, read_jsonl

import json

def test_label_to_id():
    TEST_LABELS = [
        'sadness', 'joy', 'love', 'anger', 'fear', 'surprise'
    ]

    assert label_to_id(TEST_LABELS[0]) == 0
    assert label_to_id(TEST_LABELS[1]) == 1
    assert label_to_id(TEST_LABELS[2]) == 2
    assert label_to_id(TEST_LABELS[3]) == 3
    assert label_to_id(TEST_LABELS[4]) == 4
    assert label_to_id(TEST_LABELS[5]) == 5
    assert label_to_id(None) == -1
    assert label_to_id('') == -1
    assert label_to_id("crazy") == -1



def test_build_summary():
    results = [
        {"row_id": 7, "status": "ok", "predicted_id": 0, "true_id": 1, "correct": False},
        {"row_id": 8, "status": "ok", "predicted_id": 1, "true_id": 1, "correct": True},
        {"row_id": 9, "status": "ok", "predicted_id": 0, "true_id": 3, "correct": False},
        {"row_id": 10, "status": "ok", "predicted_id": 0, "true_id": 4, "correct": False},
        {"row_id": 11, "status": "ok", "predicted_id": 2, "true_id": 0, "correct": False},
        {"row_id": 12, "status": "ok", "predicted_id": 4, "true_id": 4, "correct": True},
        {"row_id": 13, "status": "ok", "predicted_id": 1, "true_id": 1, "correct": True},
    ]

    summary = build_summary('gpt', results)

    assert summary['model'] == 'gpt'
    assert summary['size'] == 7
    assert summary['correct'] == 3
    assert summary['wrong'] == 4
    assert summary['invalid'] == 0
    assert summary['accuracy'] == round(3 * 100/ 7, 1)


def test_score_predictions(tmp_path):
    data = [
        {"text": "i feel so happy", "label": 1},
        {"text": "i am terrified", "label": 4},
        {"text": "i feel gloomy", "label": 0},
    ]

    predictions = [
        {"row_id": 2, "text": "i feel gloomy", "status": "invalid", "prediction": None, "error": "parse failed"},
        {"row_id": 0, "text": "i feel so happy", "status": "ok", "prediction": "joy", "error": None},
        {"row_id": 1, "text": "i am terrified", "status": "ok", "prediction": "sadness", "error": None},
    ]

    dataset_path = tmp_path / 'fake_dataset.csv'
    with open(dataset_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(data)

    predictions_path = tmp_path / 'fake_predictions.jsonl'
    with open(predictions_path, 'w') as f:
        for p in predictions:
            f.write(json.dumps(p) + '\n')

    results_file = score_predictions(predictions_path, dataset_path, tmp_path)
    results = read_jsonl(results_file)

    assert len(results) == 3

    assert results[0]['row_id'] == 2
    assert results[0]['status'] == 'invalid'
    assert results[0]['predicted_id'] == -1
    assert results[0]['true_id'] == 0
    assert results[0]['correct'] == False

    assert results[1]['row_id'] == 0
    assert results[1]['status'] == 'ok'
    assert results[1]['predicted_id'] == 1
    assert results[1]['true_id'] == 1
    assert results[1]['correct'] == True

    assert results[2]['row_id'] == 1
    assert results[2]['status'] == 'ok'
    assert results[2]['predicted_id'] == 0
    assert results[2]['true_id'] == 4
    assert results[2]['correct'] == False



